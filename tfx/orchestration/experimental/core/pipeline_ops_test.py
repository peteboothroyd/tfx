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
"""Tests for tfx.orchestration.experimental.core.pipeline_ops."""

import base64
import copy
import os
import threading
import time

from absl.testing.absltest import mock
import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import async_pipeline_task_gen
from tfx.orchestration.experimental.core import pipeline_ops
from tfx.orchestration.experimental.core import status as status_lib
from tfx.orchestration.experimental.core import sync_pipeline_task_gen
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.portable import test_utils as tu
from tfx.proto.orchestration import pipeline_pb2

from ml_metadata.proto import metadata_store_pb2


def _test_pipeline(pipeline_id,
                   execution_mode: pipeline_pb2.Pipeline.ExecutionMode = (
                       pipeline_pb2.Pipeline.ASYNC)):
  pipeline = pipeline_pb2.Pipeline()
  pipeline.pipeline_info.id = pipeline_id
  pipeline.execution_mode = execution_mode
  return pipeline


class PipelineOpsTest(tu.TfxTest):

  def setUp(self):
    super(PipelineOpsTest, self).setUp()
    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())

    # Makes sure multiple connections within a test always connect to the same
    # MLMD instance.
    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    self._metadata_path = metadata_path
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    self._mlmd_connection = metadata.Metadata(
        connection_config=connection_config)

  def test_initiate_pipeline_start(self):
    with self._mlmd_connection as m:
      # Initiate a pipeline start.
      pipeline1 = _test_pipeline('pipeline1')
      status_or_execution = pipeline_ops.initiate_pipeline_start(m, pipeline1)
      self.assertIsInstance(status_or_execution, metadata_store_pb2.Execution)

      # Initiate another pipeline start.
      pipeline2 = _test_pipeline('pipeline2')
      status_or_execution = pipeline_ops.initiate_pipeline_start(m, pipeline2)
      self.assertIsInstance(status_or_execution, metadata_store_pb2.Execution)

      # No error raised => context/execution types exist.
      m.store.get_context_type(pipeline_ops._ORCHESTRATOR_RESERVED_ID)
      m.store.get_execution_type(pipeline_ops._ORCHESTRATOR_RESERVED_ID)

      # Verify MLMD state.
      contexts = m.store.get_contexts_by_type(
          pipeline_ops._ORCHESTRATOR_RESERVED_ID)
      self.assertLen(contexts, 2)
      self.assertCountEqual([
          pipeline_ops._orchestrator_context_name(
              task_lib.PipelineUid.from_pipeline(pipeline1)),
          pipeline_ops._orchestrator_context_name(
              task_lib.PipelineUid.from_pipeline(pipeline2))
      ], [c.name for c in contexts])

      for context in contexts:
        executions = m.store.get_executions_by_context(context.id)
        self.assertLen(executions, 1)
        self.assertEqual(metadata_store_pb2.Execution.NEW,
                         executions[0].last_known_state)
        retrieved_pipeline = pipeline_pb2.Pipeline()
        retrieved_pipeline.ParseFromString(
            base64.b64decode(executions[0].custom_properties[
                pipeline_ops._PIPELINE_IR].string_value))
        expected_pipeline_id = (
            pipeline_ops._pipeline_uid_from_context(context).pipeline_id)
        self.assertEqual(
            _test_pipeline(expected_pipeline_id), retrieved_pipeline)

  def test_initiate_pipeline_start_new_execution(self):
    with self._mlmd_connection as m:
      pipeline1 = _test_pipeline('pipeline1')
      status_or_execution = pipeline_ops.initiate_pipeline_start(m, pipeline1)
      self.assertIsInstance(status_or_execution, metadata_store_pb2.Execution)

      # Error if attempted to initiate when old one is active.
      status_or_execution = pipeline_ops.initiate_pipeline_start(m, pipeline1)
      self.assertIsInstance(status_or_execution, status_lib.Status)
      self.assertEqual(status_lib.Code.ALREADY_EXISTS, status_or_execution.code)

      # Fine to initiate after the previous one is inactive.
      executions = m.store.get_executions_by_type(
          pipeline_ops._ORCHESTRATOR_RESERVED_ID)
      self.assertLen(executions, 1)
      executions[0].last_known_state = metadata_store_pb2.Execution.COMPLETE
      m.store.put_executions(executions)
      status_or_execution = pipeline_ops.initiate_pipeline_start(m, pipeline1)
      self.assertIsInstance(status_or_execution, metadata_store_pb2.Execution)
      self.assertEqual(metadata_store_pb2.Execution.NEW,
                       status_or_execution.last_known_state)

      # Verify MLMD state.
      contexts = m.store.get_contexts_by_type(
          pipeline_ops._ORCHESTRATOR_RESERVED_ID)
      self.assertLen(contexts, 1)
      self.assertEqual(
          pipeline_ops._orchestrator_context_name(
              task_lib.PipelineUid.from_pipeline(pipeline1)), contexts[0].name)
      executions = m.store.get_executions_by_context(contexts[0].id)
      self.assertLen(executions, 2)
      self.assertCountEqual([
          metadata_store_pb2.Execution.COMPLETE,
          metadata_store_pb2.Execution.NEW
      ], [e.last_known_state for e in executions])

  def test_initiate_pipeline_stop(self):
    with self._mlmd_connection as m:
      pipeline1 = _test_pipeline('pipeline1')
      pipeline_ops.initiate_pipeline_start(m, pipeline1)
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline1)

      status_or_execution = pipeline_ops.initiate_pipeline_stop(m, pipeline_uid)
      self.assertIsInstance(status_or_execution, metadata_store_pb2.Execution)

      # Verify MLMD state.
      executions = m.store.get_executions_by_type(
          pipeline_ops._ORCHESTRATOR_RESERVED_ID)
      self.assertLen(executions, 1)
      execution = executions[0]
      self.assertEqual(
          1,
          execution.custom_properties[pipeline_ops._STOP_INITIATED].int_value)

  def test_initiate_pipeline_stop_non_existent(self):
    with self._mlmd_connection as m:
      # Initiate stop without creating one.
      status_or_execution = pipeline_ops.initiate_pipeline_stop(
          m, task_lib.PipelineUid(pipeline_id='foo', pipeline_run_id=None))
      self.assertIsInstance(status_or_execution, status_lib.Status)
      self.assertEqual(status_lib.Code.NOT_FOUND, status_or_execution.code)

      # Initiate pipeline start and mark it completed.
      pipeline1 = _test_pipeline('pipeline1')
      execution = pipeline_ops.initiate_pipeline_start(m, pipeline1)
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline1)
      self.assertIsInstance(
          pipeline_ops.initiate_pipeline_stop(m, pipeline_uid),
          metadata_store_pb2.Execution)
      execution.last_known_state = metadata_store_pb2.Execution.COMPLETE
      m.store.put_executions([execution])

      # Try to initiate stop again.
      status_or_execution = pipeline_ops.initiate_pipeline_stop(m, pipeline_uid)
      self.assertIsInstance(status_or_execution, status_lib.Status)
      self.assertEqual(status_lib.Code.NOT_FOUND, status_or_execution.code)

  def test_wait_for_inactivation(self):
    with self._mlmd_connection as m:
      pipeline1 = _test_pipeline('pipeline1')
      status_or_execution = pipeline_ops.initiate_pipeline_start(m, pipeline1)
      self.assertIsInstance(status_or_execution, metadata_store_pb2.Execution)
      execution = status_or_execution

      def _inactivate(execution):
        time.sleep(2.0)
        with pipeline_ops._PIPELINE_OPS_LOCK:
          execution.last_known_state = metadata_store_pb2.Execution.COMPLETE
          m.store.put_executions([execution])

      thread = threading.Thread(
          target=_inactivate, args=(copy.deepcopy(execution),))
      thread.start()

      pipeline_ops.wait_for_inactivation(
          m, execution, timeout_secs=5.0, wait_tick_duration_secs=0.1)

      thread.join()

  def test_wait_for_inactivation_timeout(self):
    with self._mlmd_connection as m:
      pipeline1 = _test_pipeline('pipeline1')
      status_or_execution = pipeline_ops.initiate_pipeline_start(m, pipeline1)
      self.assertIsInstance(status_or_execution, metadata_store_pb2.Execution)
      execution = status_or_execution

      with self.assertRaisesRegex(
          TimeoutError, 'Timed out.*waiting for execution inactivation.'):
        pipeline_ops.wait_for_inactivation(
            m, execution, timeout_secs=1.0, wait_tick_duration_secs=0.1)

  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  def test_generate_tasks_async_pipelines(self, mock_async_task_gen,
                                          mock_sync_task_gen):
    with self._mlmd_connection as m:
      # One active pipeline.
      pipeline1 = _test_pipeline('pipeline1')
      execution1 = pipeline_ops.initiate_pipeline_start(m, pipeline1)
      self.assertIsInstance(execution1, metadata_store_pb2.Execution)

      # Another active pipeline (with previously completed execution).
      pipeline2 = _test_pipeline('pipeline2')
      execution2 = pipeline_ops.initiate_pipeline_start(m, pipeline2)
      execution2.last_known_state = metadata_store_pb2.Execution.COMPLETE
      m.store.put_executions([execution2])
      execution2 = pipeline_ops.initiate_pipeline_start(m, pipeline2)
      self.assertIsInstance(execution2, metadata_store_pb2.Execution)

      # One inactive pipeline.
      pipeline3 = _test_pipeline('pipeline3')
      execution3 = pipeline_ops.initiate_pipeline_start(m, pipeline3)
      execution3.last_known_state = metadata_store_pb2.Execution.FAILED
      m.store.put_executions([execution3])

      # One stop initiated pipeline.
      pipeline4 = _test_pipeline('pipeline4')
      pipeline4.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline4.nodes.add().pipeline_node.node_info.id = 'Evaluator'
      pipeline_ops.initiate_pipeline_start(m, pipeline4)
      execution4 = pipeline_ops.initiate_pipeline_stop(
          m, task_lib.PipelineUid.from_pipeline(pipeline4))

      def _exec_node_tasks():
        for pipeline_id in ('pipeline1', 'pipeline2'):
          yield [
              test_utils.create_exec_node_task(
                  node_uid=task_lib.NodeUid(
                      pipeline_uid=task_lib.PipelineUid(
                          pipeline_id=pipeline_id, pipeline_run_id=None),
                      node_id='Trainer')),
              test_utils.create_exec_node_task(
                  node_uid=task_lib.NodeUid(
                      pipeline_uid=task_lib.PipelineUid(
                          pipeline_id=pipeline_id, pipeline_run_id=None),
                      node_id='Transform'))
          ]

      mock_async_task_gen.return_value.generate.side_effect = _exec_node_tasks()

      task_queue = tq.TaskQueue()
      status = pipeline_ops.generate_tasks(m, task_queue)
      self.assertEqual(status_lib.Code.OK, status.code)

      self.assertEqual(2, mock_async_task_gen.return_value.generate.call_count)
      mock_sync_task_gen.assert_not_called()

      # Verify that tasks are enqueued in the correct order:

      # First, cancellation tasks for nodes of stop initiated pipelines.
      for node_id in ('Transform', 'Evaluator'):
        task = task_queue.dequeue()
        task_queue.task_done(task)
        self.assertTrue(task_lib.is_cancel_node_task(task))
        self.assertEqual(node_id, task.node_uid.node_id)

      # Next, pipeline unregistration task for the stop initiated pipeline.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(task_lib.is_unregister_pipeline_task(task))
      self.assertEqual('pipeline4', task.pipeline_uid.pipeline_id)

      # The unregistration callback should flip the execution state in MLMD.
      task.callback()
      [execution] = m.store.get_executions_by_id([execution4.id])
      self.assertEqual(metadata_store_pb2.Execution.CANCELED,
                       execution.last_known_state)

      # Next, pipeline registration task and exec tasks for the two active
      # pipelines.
      actives = set(['pipeline1', 'pipeline2'])
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(task_lib.is_register_pipeline_task(task))
      pipeline_id = task.pipeline_uid.pipeline_id
      self.assertIn(pipeline_id, actives)
      for _ in range(2):
        task = task_queue.dequeue()
        task_queue.task_done(task)
        self.assertTrue(task_lib.is_exec_node_task(task))
        self.assertEqual(pipeline_id, task.node_uid.pipeline_uid.pipeline_id)

      actives.remove(pipeline_id)

      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(task_lib.is_register_pipeline_task(task))
      pipeline_id = task.pipeline_uid.pipeline_id
      self.assertIn(pipeline_id, actives)
      for _ in range(2):
        task = task_queue.dequeue()
        task_queue.task_done(task)
        self.assertTrue(task_lib.is_exec_node_task(task))
        self.assertEqual(pipeline_id, task.node_uid.pipeline_uid.pipeline_id)

      # All tasks consumed.
      self.assertTrue(task_queue.is_empty())


if __name__ == '__main__':
  tf.test.main()
