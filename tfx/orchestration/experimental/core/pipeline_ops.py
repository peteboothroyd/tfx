# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pipeline-level operations."""

import base64
import copy
import functools
import threading
import time
from typing import List, Optional, Text, Union

from absl import logging
import attr
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import async_pipeline_task_gen
from tfx.orchestration.experimental.core import status as status_lib
from tfx.orchestration.experimental.core import sync_pipeline_task_gen
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.portable.mlmd import common_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2

from ml_metadata.proto import metadata_store_pb2

_ORCHESTRATOR_RESERVED_ID = '__ORCHESTRATOR__'
_PIPELINE_IR = 'pipeline_ir'
_STOP_INITIATED = 'stop_initiated'
_ORCHESTRATOR_EXECUTION_TYPE = metadata_store_pb2.ExecutionType(
    name=_ORCHESTRATOR_RESERVED_ID)

# A coarse grained lock is used to ensure serialization of pipeline operations
# since there isn't a suitable MLMD transaction API.
_PIPELINE_OPS_LOCK = threading.RLock()


def _pipeline_ops_lock(fn):
  """Decorator to run `fn` within `_PIPELINE_OPS_LOCK` context."""

  @functools.wraps(fn)
  def _wrapper(*args, **kwargs):
    with _PIPELINE_OPS_LOCK:
      return fn(*args, **kwargs)

  return _wrapper


@_pipeline_ops_lock
def initiate_pipeline_start(
    mlmd_handle: metadata.Metadata, pipeline: pipeline_pb2.Pipeline
) -> Union[status_lib.Status, metadata_store_pb2.Execution]:
  """Initiates a pipeline start operation.

  Upon success, MLMD is updated to signal that the given pipeline must be
  started.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline: IR of the pipeline to start.

  Returns:
    The pipeline-level MLMD execution proto upon success, otherwise a status
    indicating any errors.
  """
  pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline)
  context = context_lib.register_context_if_not_exists(
      mlmd_handle,
      context_type_name=_ORCHESTRATOR_RESERVED_ID,
      context_name=_orchestrator_context_name(pipeline_uid))

  executions = mlmd_handle.store.get_executions_by_context(context.id)
  if any(e for e in executions if execution_lib.is_execution_active(e)):
    return status_lib.Status(
        code=status_lib.Code.ALREADY_EXISTS,
        message=f'Pipeline with uid {pipeline_uid} already started.')

  execution = execution_lib.prepare_execution(
      mlmd_handle,
      _ORCHESTRATOR_EXECUTION_TYPE,
      metadata_store_pb2.Execution.NEW,
      exec_properties={
          _PIPELINE_IR:
              base64.b64encode(pipeline.SerializeToString()).decode('utf-8')
      })
  execution = execution_lib.put_execution(mlmd_handle, execution, [context])
  logging.info('Registered execution (id: %s) for the pipeline with uid: %s',
               execution.id, pipeline_uid)
  return execution


@_pipeline_ops_lock
def initiate_pipeline_stop(
    mlmd_handle: metadata.Metadata, pipeline_uid: task_lib.PipelineUid
) -> Union[status_lib.Status, metadata_store_pb2.Execution]:
  """Initiates a pipeline stop operation.

  Upon success, MLMD is updated to signal that the pipeline given by
  `pipeline_uid` must be stopped.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline_uid: Uid of the pipeline to be stopped.

  Returns:
    The pipeline-level MLMD execution proto upon success, otherwise a status
    indicating any errors.
  """
  context = mlmd_handle.store.get_context_by_type_and_name(
      type_name=_ORCHESTRATOR_RESERVED_ID,
      context_name=_orchestrator_context_name(pipeline_uid))
  if not context:
    return status_lib.Status(
        code=status_lib.Code.NOT_FOUND,
        message=f'No active pipeline with uid {pipeline_uid} to stop.')

  executions = mlmd_handle.store.get_executions_by_context(context.id)
  active_executions = [
      e for e in executions if execution_lib.is_execution_active(e)
  ]
  if not active_executions:
    return status_lib.Status(
        code=status_lib.Code.NOT_FOUND,
        message=f'No active pipeline with uid {pipeline_uid} to stop.')
  if len(active_executions) > 1:
    return status_lib.Status(
        code=status_lib.Code.FAILED_PRECONDITION,
        message=(f'Error while stopping pipeline uid {pipeline_uid}: found '
                 f'{len(active_executions)} active executions, expected 1.'))
  execution = active_executions[0]
  _set_stop_initiated_property(execution)
  mlmd_handle.store.put_executions([execution])
  return execution


_DEFAULT_WAIT_TICK_DURATION_SECS = 10.0
_DEFAULT_WAIT_FOR_INACTIVATION_TIMEOUT_SECS = 120.0


def wait_for_inactivation(
    mlmd_handle: metadata.Metadata,
    execution: metadata_store_pb2.Execution,
    timeout_secs: float = _DEFAULT_WAIT_FOR_INACTIVATION_TIMEOUT_SECS,
    wait_tick_duration_secs: float = _DEFAULT_WAIT_TICK_DURATION_SECS) -> None:
  """Waits for the given execution to become inactive.

  Args:
    mlmd_handle: A handle to the MLMD db.
    execution: Execution whose inactivation is waited.
    timeout_secs: Amount of time in seconds to wait.
    wait_tick_duration_secs: Determines how often to poll MLMD.

  Raises:
    TimeoutError: If execution is not inactive after waiting `timeout_secs`.
  """
  time_budget = timeout_secs
  while time_budget > 0.0:
    with _PIPELINE_OPS_LOCK:
      updated_executions = mlmd_handle.store.get_executions_by_id(
          [execution.id])
    if not execution_lib.is_execution_active(updated_executions[0]):
      return
    time.sleep(wait_tick_duration_secs)
    time_budget -= wait_tick_duration_secs
  raise TimeoutError(
      f'Timed out ({timeout_secs} secs) waiting for execution inactivation.')


@attr.s
class _PipelineDetail:
  """Data class helper for `generate_tasks`."""
  context = attr.ib(type=metadata_store_pb2.Context)
  execution = attr.ib(type=metadata_store_pb2.Execution)
  pipeline = attr.ib(type=pipeline_pb2.Pipeline)
  stop_initiated = attr.ib(type=bool, default=False)
  generator = attr.ib(type=Optional[task_gen.TaskGenerator], default=None)


@_pipeline_ops_lock
def generate_tasks(mlmd_handle: metadata.Metadata,
                   task_queue: tq.TaskQueue) -> status_lib.Status:
  """Generates and enqueues tasks to be performed.

  Embodies the core functionality of the main orchestration loop that scans MLMD
  pipeline execution states, generates and enqueues the tasks to be performed.

  Args:
    mlmd_handle: A handle to the MLMD db.
    task_queue: A `TaskQueue` instance into which any tasks will be enqueued.

  Returns:
    Operation status.
  """
  contexts = mlmd_handle.store.get_contexts_by_type(_ORCHESTRATOR_RESERVED_ID)
  if not contexts:
    logging.info('No active pipelines to run.')
    return status_lib.Status(code=status_lib.Code.OK)

  pipeline_details = []
  for context in contexts:
    active_executions = [
        e for e in mlmd_handle.store.get_executions_by_context(context.id)
        if execution_lib.is_execution_active(e)
    ]
    if len(active_executions) > 1:
      return status_lib.Status(
          code=status_lib.Code.FAILED_PRECONDITION,
          message=(f'Expected 1 but found {len(active_executions)} active '
                   f'executions for context named: {context.name}'))
    if not active_executions:
      continue
    execution = active_executions[0]

    # TODO(goutham): Instead of parsing the pipeline IR each time, we could
    # cache the parsed pipeline IR and only parse again if last update time of
    # the execution is different.
    pipeline_ir_b64 = common_utils.get_metadata_value(
        execution.custom_properties[_PIPELINE_IR])
    pipeline = pipeline_pb2.Pipeline()
    pipeline.ParseFromString(base64.b64decode(pipeline_ir_b64))

    stop_initiated = _is_stop_initiated(execution)

    if not stop_initiated:
      if pipeline.execution_mode == pipeline_pb2.Pipeline.SYNC:
        generator = sync_pipeline_task_gen.SyncPipelineTaskGenerator(
            mlmd_handle, pipeline, task_queue.contains_task_id)
      elif pipeline.execution_mode == pipeline_pb2.Pipeline.ASYNC:
        generator = async_pipeline_task_gen.AsyncPipelineTaskGenerator(
            mlmd_handle, pipeline, task_queue.contains_task_id)
      else:
        return status_lib.Status(
            code=status_lib.Code.FAILED_PRECONDITION,
            message=(
                f'Only SYNC and ASYNC pipeline execution modes supported; '
                f'found pipeline with execution mode: {pipeline.execution_mode}'
            ))
    else:
      generator = None

    pipeline_details.append(
        _PipelineDetail(
            context=context,
            execution=execution,
            pipeline=pipeline,
            stop_initiated=stop_initiated,
            generator=generator))

  # The below operations are ordered so that the tasks added to the task queue
  # make sense both when the task queue is already filled from a previous
  # iteration as well as when the task queue is empty (in the event of an
  # orchestrator restart). This property is important to maintain when making
  # any future changes.

  # Enqueue cancellation tasks for all nodes for pipelines for which stopping
  # has been initiated. For nodes that are not scheduled, cancellation tasks are
  # ignored by the task manager. Follow by unregistering the pipeline.
  for detail in (d for d in pipeline_details if d.stop_initiated):
    nodes = _get_all_pipeline_nodes(detail.pipeline)
    for node in nodes:
      task_queue.enqueue(
          task_lib.CancelNodeTask(
              node_uid=task_lib.NodeUid.from_pipeline_node(
                  detail.pipeline, node)))
    # Flip the execution state to "CANCELED" when unregistered.
    updated_execution = copy.deepcopy(detail.execution)
    updated_execution.last_known_state = metadata_store_pb2.Execution.CANCELED
    task_queue.enqueue(
        task_lib.UnregisterPipelineTask(
            pipeline_uid=task_lib.PipelineUid.from_pipeline(detail.pipeline),
            callback=functools.partial(_put_execution, mlmd_handle,
                                       updated_execution)))

  # Enqueue pipeline registration tasks for pipelines that are active. If a
  # pipeline is already registered (from a previous iteration), task manager
  # treats it as a no-op, so it's safe to create duplicate registration tasks.
  # Then, for each active pipeline, generate and enqueue node execution tasks.
  for detail in (d for d in pipeline_details if not d.stop_initiated):
    # Flip the execution state to "RUNNING" upon registration.
    updated_execution = copy.deepcopy(detail.execution)
    updated_execution.last_known_state = metadata_store_pb2.Execution.RUNNING
    task_queue.enqueue(
        task_lib.RegisterPipelineTask(
            pipeline_uid=task_lib.PipelineUid.from_pipeline(detail.pipeline),
            pipeline=detail.pipeline,
            callback=functools.partial(_put_execution, mlmd_handle,
                                       updated_execution)))
    # TODO(goutham): Consider concurrent task generation.
    tasks = detail.generator.generate()
    for task in tasks:
      task_queue.enqueue(task)

  return status_lib.Status(code=status_lib.Code.OK)


def _orchestrator_context_name(pipeline_uid: task_lib.PipelineUid) -> Text:
  """Returns orchestrator reserved context name."""
  result = f'{_ORCHESTRATOR_RESERVED_ID}_{pipeline_uid.pipeline_id}'
  if pipeline_uid.pipeline_run_id:
    result = f'{result}:{pipeline_uid.pipeline_run_id}'
  return result


def _pipeline_uid_from_context(
    context: metadata_store_pb2.Context) -> task_lib.PipelineUid:
  """Returns pipeline uid from orchestrator reserved context."""
  suffix = context.name.split(_ORCHESTRATOR_RESERVED_ID + '_')[1]
  parts = suffix.split(':')
  pipeline_id = parts[0]
  if len(parts) == 2:
    pipeline_run_id = parts[1]
  else:
    pipeline_run_id = None
  return task_lib.PipelineUid(
      pipeline_id=pipeline_id, pipeline_run_id=pipeline_run_id)


def _set_stop_initiated_property(
    execution: metadata_store_pb2.Execution) -> None:
  common_utils.set_metadata_value(execution.custom_properties[_STOP_INITIATED],
                                  1)


def _is_stop_initiated(execution: metadata_store_pb2.Execution) -> bool:
  return common_utils.get_metadata_value(
      execution.custom_properties[_STOP_INITIATED]) == 1


def _get_all_pipeline_nodes(
    pipeline: pipeline_pb2.Pipeline) -> List[pipeline_pb2.PipelineNode]:
  """Returns all pipeline nodes in the given pipeline."""
  result = []
  for pipeline_or_node in pipeline.nodes:
    which = pipeline_or_node.WhichOneof('node')
    # TODO(goutham): Handle sub-pipelines.
    if which == 'pipeline_node':
      result.append(pipeline_or_node.pipeline_node)
    else:
      raise NotImplementedError('Only pipeline nodes supported.')
  return result


@_pipeline_ops_lock
def _put_execution(mlmd_handle: metadata.Metadata,
                   execution: metadata_store_pb2.Execution) -> None:
  """Puts execution to MLMD within pipeline operations lock context."""
  mlmd_handle.store.put_executions([execution])
