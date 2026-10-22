"""Exercise frontend event logic in Node with a temporary canvas, not a browser."""
import ast
import shutil
import subprocess
from pathlib import Path

import pytest


def test_two_click_coordinates_and_frame_isolation(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for canvas event checks")
    source = Path(__file__).parents[1] / "playlab/box_drawer.py"
    parsed = ast.parse(source.read_text())
    javascript = next(ast.literal_eval(item.value) for item in parsed.body
                      if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DRAW_JS"
                                                             for t in item.targets))
    harness = """
    import assert from 'node:assert/strict';
    class Image {
      naturalWidth=1000; naturalHeight=500;
      set src(value) { this.onload?.(); }
    }
    const context = new Proxy({}, {get:(target,key)=>target[key]??(()=>{})});
    const canvas = {getContext:()=>context, focus:()=>{},
      getBoundingClientRect:()=>({left:10,top:20,width:500,height:250})};
    const status = {};
    const parent = {querySelector:key=>key==='canvas'?canvas:status};
    const events = [];
    const component = {parentElement:parent, data:{identity:'frame-A',image:'fixture',boxes:[]},
      setTriggerValue:(key,value)=>events.push([key,value])};
    drawCanvas(component);
    const click = (x,y)=>canvas.onpointerdown({clientX:x,clientY:y,preventDefault:()=>{}});
    click(410,220);
    assert.equal(events.length,0,'first click must not emit a box');
    click(110,70);
    assert.deepEqual(events,[['box',[.2,.2,.8,.8]]]);
    click(260,145);
    canvas.onkeydown({key:'Escape'});
    click(110,70);
    assert.equal(events.length,1,'Escape must clear the pending first corner');
    component.data.identity='frame-B';
    drawCanvas(component);
    click(110,70);
    assert.equal(events.length,1,'pending corners must not leak across frames');
    click(110,70);
    assert.equal(events.length,1,'zero-area boxes must not be emitted');
    canvas.getBoundingClientRect=()=>({left:10,top:20,width:1000,height:500});
    click(810,420); click(210,120);
    assert.deepEqual(events[1],['box',[.2,.2,.8,.8]],'CSS resizing must preserve source coordinates');
    """
    module = tmp_path / "drawer-check.mjs"
    module.write_text(javascript + harness)
    subprocess.run([node, str(module)], check=True, capture_output=True, text=True, timeout=15)
