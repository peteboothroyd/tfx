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
"""Experimental Resolver for getting the oldest artifact."""

from typing import Dict, List, Optional, Text

from tfx import types
from tfx.dsl.resolvers import base_resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata


def _time_order(artifact: types.Artifact):
  # Use MLMD Artifact.id for tie breaking so that result is deterministic.
  return (artifact.mlmd_artifact.last_update_time_since_epoch, artifact.id)


class OldestArtifactsResolver(base_resolver.BaseResolver):
  """Resolver that returns the oldest artifact for each input channel."""

  def resolve(
      self,
      pipeline_info: data_types.PipelineInfo,
      metadata_handler: metadata.Metadata,
      source_channels: Dict[Text, types.Channel],
  ) -> base_resolver.ResolveResult:
    raise NotImplementedError('Asynchronous mode only resolver')

  def resolve_artifacts(
      self, context: base_resolver.ResolverContext,
      input_dict: Dict[Text, List[types.Artifact]]
  ) -> Optional[Dict[Text, List[types.Artifact]]]:
    result = {}
    for input_key, artifacts in input_dict.items():
      if not artifacts:
        return None
      result[input_key] = [min(artifacts, key=_time_order)]
    return result
