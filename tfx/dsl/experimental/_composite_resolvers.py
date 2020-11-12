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
"""Internal composite resolver reprentations."""

from tfx.dsl.resolvers import base_resolver


class LatestUnprocessedArtifactsResolver(base_resolver.BaseResolver):
  """Resolve latest artifacts that are not processed by the input component."""


class OldestUnprocessedArtifactsResolver(base_resolver.BaseResolver):
  """Resolve oldest artifacts that are not processed by the input component."""
