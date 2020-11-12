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
"""Experimental Resolver for getting the unprocessed artifact."""

from typing import Dict, Text, List, Optional

from tfx import types
from tfx.dsl.resolvers import base_resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration.portable.mlmd import event_lib
from tfx.orchestration.portable.mlmd import execution_lib


class UnprocessedArtifactsResolver(base_resolver.BaseResolver):
  """Resolver that filters out already processed artifacts.

  An input Artifact is considered "processed" if
  (1) it has an associated input event with the same input key
  (2) whose associated execution was successful
  (3) and the execution belongs to the current pipeline node contexts.
  """

  def resolve(
      self,
      pipeline_info: data_types.PipelineInfo,
      metadata_handler: metadata.Metadata,
      source_channels: Dict[Text, types.Channel],
  ) -> base_resolver.ResolveResult:
    raise NotImplementedError()

  def resolve_artifacts(
      self, context: base_resolver.ResolverContext,
      input_dict: Dict[Text, List[types.Artifact]]
  ) -> Optional[Dict[Text, List[types.Artifact]]]:
    # (3) Find executions that belong to the current pipeline node contexts.
    executions = execution_lib.get_executions_associated_with_all_contexts(
        context.metadata_handler,
        context.pipeline_node_contexts)
    # (2) whose execution was successful
    execution_ids = [e.id for e in executions
                     if execution_lib.is_execution_successful(e)]
    events = context.store.get_events_by_execution_ids(execution_ids)
    event_by_artifact_id = {
        event.artfiact_id: event
        for event in events
    }

    def is_processed_before(artifact, input_key):
      # (1) whose associated event is an input event of the same input key.
      return (artifact.id in event_by_artifact_id
              and event_lib.is_valid_input_event(
                  event_by_artifact_id[artifact.id], input_key))

    result = {}
    for input_key, artifacts in input_dict.items():
      for artifact in artifacts:
        if not is_processed_before(artifact, input_key):
          result.setdefault(input_key, []).append(artifact)
    return result
