#
# Copyright (C) 2017 - Massachusetts Institute of Technology (MIT)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from .base import ImageEffect


class SanityCheck(ImageEffect):
    """
    Ensure that there are no negative values.
    """
    version = 0.1
    units = ''

    def __init__(self):
        super(SanityCheck, self).__init__()

    def apply_to_image(self, image, header={}):
        # FIXME: log/notify when negative values encountered
        image[image < 0.0] = 0.0 # ensure that there are no negative values
        return image