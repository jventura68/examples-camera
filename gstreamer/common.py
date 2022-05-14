# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utilities."""
import collections
import io
import time

ALPHA_LOW = 0.3
ALPHA_HIGH = 0.8

NOT_DETECT_COLOR = "red"
DETECT_COLOR = "green"
MOVE_COLOR = "red"
BACKGROUND_COLOR = "gray"
SVG_HEADER = '<svg width="{w}" height="{h}" version="1.1" >'
SVG_RECT = '<rect x="{x}" y="{y}" width="{w}" height="{h}" stroke="{s}" stroke-width="{sw}" fill="none" />'
SVG_TEXT = '''
<text x="{x}" y="{y}" font-size="{fs}" dx="0.05em" dy="0.05em" fill="black">{t}</text>
<text x="{x}" y="{y}" font-size="{fs}" fill="white">{t}</text>
'''
SVG_R_ARROW = '''
<polygon points="{x},{y} {x2},{center} {x},{y2}" 
         stroke="none" 
         fill="green" 
         style="fill-opacity: {alpha};">
</polygon>'''

SVG_L_ARROW = '''
<polygon points="{x},{y} {x2},{center} {x},{y2}" 
         stroke="none" 
         fill="green" 
         style="fill-opacity: {alpha};">
</polygon>'''

SVG_OK = '''
<rect x="{x}" y="{y}" width="{w}" height="{h}" 
      stroke="none" fill="{fill}" 
      style="fill-opacity: {alpha};"/>
'''
SVG_FOOTER = '</svg>'


def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)

class SVG:

    xy_control ={
        "sep":20,
        "h": 40,
        "w": 40,
        "arrow_w": 60
    }
    size = None

    def __init__(self, size):
        self.io = io.StringIO()
        self.io.write(SVG_HEADER.format(w=size[0] , h=size[1]))
        self.size = size

    def add_rect(self, x, y, w, h, stroke, stroke_width):
        self.io.write(SVG_RECT.format(x=x, y=y, w=w, h=h, s=stroke, sw=stroke_width))

    def add_text(self, x, y, text, font_size):
        self.io.write(SVG_TEXT.format(x=x, y=y, t=text, fs=font_size))

    def _add_right_arrow(self, x, y, w, h, alpha):
        x2=x+w
        y2=y+h
        center = y+h/2
        self.io.write(SVG_R_ARROW.format(x=x, y=y, x2=x2, y2=y2, center=center, alpha=alpha))

    def _add_left_arrow(self, x, y, w, h, alpha):
        x2=x-w
        y2=y+h
        center = y+h/2
        self.io.write(SVG_L_ARROW.format(x=x, y=y, x2=x2, y2=y2, center = center, alpha=alpha))

    def _block(self, x, y, w, h, fill, alpha):
        x2=x+w
        y2=y+h
        self.io.write(SVG_OK.format(x=x, y=y, w=w, h=h, fill=fill, alpha=alpha))

    def add_controls (self, left=False, cam_ok=False, right=False, bounds=None):
        center = self.size[0]/2
        y = self.size[1] - self.xy_control["sep"] - self.xy_control["h"]
        x_arrow_left = center - self.xy_control["w"]
        x_arrow_right  = center + self.xy_control["w"]
        x_cam_ok = center - self.xy_control["sep"]

        alpha_left = ALPHA_HIGH if left else ALPHA_LOW
        alpha_right = ALPHA_HIGH if right else ALPHA_LOW
        alpha_cam_ok = ALPHA_HIGH *2/3 if cam_ok else ALPHA_HIGH
        cam_fill = DETECT_COLOR if cam_ok else NOT_DETECT_COLOR


        self._add_left_arrow(x=x_arrow_left, 
                            y=y, 
                            w=self.xy_control["arrow_w"],
                            h=self.xy_control["h"], 
                            alpha=alpha_left)

        self._block(x=x_cam_ok,
                    y=y,
                    w=self.xy_control["w"],
                    h=self.xy_control["w"],
                    fill=cam_fill,
                    alpha=alpha_cam_ok)

        self._add_right_arrow(x=x_arrow_right, 
                            y=y, 
                            w=self.xy_control["arrow_w"],
                            h=self.xy_control["h"], 
                            alpha=alpha_right)
        if bounds:
            self._block(x=0,
                        y=0,
                        w=bounds[0],
                        h=self.size[1],
                        fill=BACKGROUND_COLOR,
                        alpha=0.1)
            self._block(x=bounds[1],
                        y=0,
                        w=self.size[0]-bounds[1],
                        h=self.size[1],
                        fill=BACKGROUND_COLOR,
                        alpha=0.1)


    def finish(self):
        self.io.write(SVG_FOOTER)
        return self.io.getvalue()
