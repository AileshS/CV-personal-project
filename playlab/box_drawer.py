"""Two-click annotation drafts. Drawing never writes labels or review flags."""
import base64
import io

import streamlit as st
from PIL import Image

DRAW_JS = """
export default function drawCanvas(component) {
  const {parentElement, data, setTriggerValue} = component;
  const canvas = parentElement.querySelector('canvas');
  const status = parentElement.querySelector('p');
  const state = parentElement._draftState ??= {};
  const generation = state.generation = (state.generation ?? 0) + 1;
  let loaded = false;
  if (state.identity !== data.identity) { state.point = null; state.identity = data.identity; }
  const image = new Image();
  function paint() {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0);
    for (const item of data.boxes) {
      const [x1,y1,x2,y2] = item.box;
      ctx.strokeStyle = item.draft ? '#00d5ff' : item.proposal ? '#ff8d54' : '#ffbd45';
      ctx.lineWidth = 3;
      ctx.setLineDash(item.proposal ? [8,5] : []);
      ctx.strokeRect(x1*canvas.width, y1*canvas.height, (x2-x1)*canvas.width, (y2-y1)*canvas.height);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.font = '16px sans-serif';
      ctx.fillText((item.draft ? 'Draft ' : item.proposal ? 'Suggestion ' : '') + item.label,
                   x1*canvas.width, Math.max(18, y1*canvas.height-5));
    }
    ctx.setLineDash([]);
    if (state.point) {
      ctx.fillStyle = '#00d5ff'; ctx.beginPath();
      ctx.arc(state.point[0]*canvas.width, state.point[1]*canvas.height, 5, 0, 2*Math.PI); ctx.fill();
    }
    status.textContent = state.point ? 'Now click the opposite corner. Escape cancels.' :
      'Click two opposite corners around the object. This only changes the draft.';
  }
  image.onload = () => {if(state.generation !== generation)return;
    loaded=true;canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;paint();};
  image.src = data.image;
  canvas.onpointerdown = (event) => {
    if (!loaded) return;
    event.preventDefault(); canvas.focus();
    const rect = canvas.getBoundingClientRect();
    const point = [Math.max(0, Math.min(1, (event.clientX-rect.left)/rect.width)),
                   Math.max(0, Math.min(1, (event.clientY-rect.top)/rect.height))];
    if (!state.point) {state.point=point; paint(); return;}
    const box = [Math.min(state.point[0],point[0]), Math.min(state.point[1],point[1]),
                 Math.max(state.point[0],point[0]), Math.max(state.point[1],point[1])];
    state.point=null;
    if (box[0]<box[2] && box[1]<box[3]) setTriggerValue('box', box);
    paint();
  };
  canvas.onkeydown = event => {if(event.key==='Escape'){state.point=null;paint();}};
  return () => {canvas.onpointerdown=null;canvas.onkeydown=null;image.onload=null;};
}
"""

_DRAWER = st.components.v2.component(
    "basketball_two_corner_box",
    html="""
    <canvas tabindex="0" role="img" aria-label="Basketball frame: click two opposite corners to draft a box"
      style="width:100%;height:auto;cursor:crosshair;touch-action:none;display:block"></canvas>
    <p aria-live="polite"></p>
    """, js=DRAW_JS)


def draw_frame(pixels, boxes, *, identity, key, on_box_change):
    image = Image.fromarray(pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return _DRAWER(data={"image": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(),
                         "boxes": boxes, "identity": identity}, key=key, on_box_change=on_box_change)
