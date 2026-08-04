# coding=utf-8
# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

from robodojo_eval.deploy_policy import _to_hwc_uint8_rgb


class ImageConversionTest(unittest.TestCase):
    def test_converts_chw_rgba_to_hwc_rgb(self):
        image = np.stack(
            [
                np.full((2, 5), 10, dtype=np.uint8),
                np.full((2, 5), 20, dtype=np.uint8),
                np.full((2, 5), 30, dtype=np.uint8),
                np.full((2, 5), 255, dtype=np.uint8),
            ]
        )

        converted = _to_hwc_uint8_rgb(image)

        self.assertEqual(converted.shape, (2, 5, 3))
        np.testing.assert_array_equal(converted[0, 0], [10, 20, 30])

    def test_rejects_unsupported_channel_count(self):
        image = np.zeros((2, 5, 2), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "Expected 1, 3, or 4 image channels"):
            _to_hwc_uint8_rgb(image)


if __name__ == "__main__":
    unittest.main()
