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
<rect x="{x}" y="{y}" width="{w}" height="{w}" 
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

    def _add_rigth_arrow(self, x, y, w, h, alpha):
        x2=x+w
        y2=y+h
        center = y+h/2
        self.io.write(SVG_R_ARROW.format(x=x, y=y, x2=x2, y2=y2, center=center, alpha=alpha))

    def _add_left_arrow(self, x, y, w, h, alpha):
        x2=x-w
        y2=y+h
        center = y+h/2
        self.io.write(SVG_R_ARROW.format(x=x, y=y, x2=x2, y2=y2, center=center, alpha=alpha))

    def _cam_ok(self, x, y, w, fill, alpha):
        x2=x+w
        y2=y+w
        self.io.write(SVG_OK.format(x=x, y=y, w=40, fill=fill, alpha=alpha))

    def add_controls (self, left=False, cam_ok=False, right=False):
        center = self.size[0]/2
        ymin = self.size[1] - self.xy_control["sep"] - self.xy_control["h"]
        ymax = ymin + self.xy_control["h"]
        arrow_w = self.xy_control["w"]

        alpha_left = 0.5 if left else 0.1
        alpha_right = 0.5 if right else 0.1
        alpha_cam_ok = 0.5 if cam_ok else 0.1

        self._add_left_arrow(x=center-self.xy_control["sep"], 
                            y=ymin, 
                            w=self.xy_control["arrow_w"],
                            h=self.xy_control["h"], 
                            alpha=alpha_left)

        self._cam_ok(x=center-self.xy_control["sep"],
                    y=ymin,
                    w=self.xy_control["w"],
                    fill="green",
                    alpha=alpha_cam_ok)

        self._add_left_arrow(x=center-self.xy_control["sep"], 
                            y=ymin, 
                            w=self.xy_control["arrow_w"],
                            h=self.xy_control["h"], 
                            alpha=alpha_left)

    def finish(self):
        self.io.write(SVG_FOOTER)
        return self.io.getvalue()