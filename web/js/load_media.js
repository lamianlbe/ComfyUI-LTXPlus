import { app } from "/scripts/app.js";

const NODE_TYPE = "LTXVideoLoadMedia";

/**
 * Fetch a preview frame from the backend and display it on the node.
 */
async function updatePreview(node) {
  const mediaWidget = node.widgets?.find((w) => w.name === "media");
  const frameWidget = node.widgets?.find((w) => w.name === "frame_id");
  const bypassWidget = node.widgets?.find((w) => w.name === "bypass");

  if (!mediaWidget) return;

  const filename = mediaWidget.value;
  if (!filename || filename === "none") {
    node.imgs = null;
    app.graph.setDirtyCanvas(true, true);
    return;
  }

  if (bypassWidget?.value === true) {
    node.imgs = null;
    app.graph.setDirtyCanvas(true, true);
    return;
  }

  const frameId = frameWidget?.value ?? 0;

  const url =
    `/ltxplus/preview_frame?filename=${encodeURIComponent(filename)}` +
    `&frame_id=${frameId}&type=input&t=${Date.now()}`;

  try {
    const img = new Image();
    img.onload = () => {
      node.imgs = [img];
      node.imageIndex = 0;
      requestAnimationFrame(() => {
        node.setSizeForImage?.();
        app.graph.setDirtyCanvas(true, true);
      });
    };
    img.onerror = () => {
      node.imgs = null;
      app.graph.setDirtyCanvas(true, true);
    };
    img.src = url;
  } catch (e) {
    console.warn("LTXVideoLoadMedia: preview fetch failed", e);
  }
}

/**
 * Fetch and display frame count info on the node.
 */
async function updateFrameCount(node) {
  const mediaWidget = node.widgets?.find((w) => w.name === "media");
  if (!mediaWidget) return;

  const filename = mediaWidget.value;
  if (!filename || filename === "none") {
    node._ltx_frame_count = null;
    return;
  }

  try {
    const resp = await fetch(
      `/ltxplus/frame_count?filename=${encodeURIComponent(filename)}&type=input`
    );
    const data = await resp.json();
    node._ltx_frame_count = data.frame_count;
  } catch (e) {
    node._ltx_frame_count = null;
  }
}

app.registerExtension({
  name: "LTXPlus.LoadMedia",

  async nodeCreated(node) {
    if (node.comfyClass !== NODE_TYPE) return;

    // Debounce timer for preview updates
    let previewTimer = null;
    const schedulePreview = () => {
      if (previewTimer) clearTimeout(previewTimer);
      previewTimer = setTimeout(() => updatePreview(node), 200);
    };

    // Hook into widget value changes
    for (const w of node.widgets || []) {
      if (w.name === "media") {
        const origCallback = w.callback;
        w.callback = function (...args) {
          origCallback?.apply(this, args);
          updateFrameCount(node);
          schedulePreview();
        };
        // Initial load
        setTimeout(() => {
          updateFrameCount(node);
          updatePreview(node);
        }, 500);
      } else if (w.name === "frame_id") {
        const origCallback = w.callback;
        w.callback = function (...args) {
          origCallback?.apply(this, args);
          schedulePreview();
        };
      } else if (w.name === "bypass") {
        const origCallback = w.callback;
        w.callback = function (...args) {
          origCallback?.apply(this, args);
          schedulePreview();
        };
      }
    }

    // Draw frame count badge on the node
    const origDrawForeground = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
      origDrawForeground?.apply(this, arguments);

      if (this._ltx_frame_count != null && this._ltx_frame_count > 1) {
        const frameWidget = this.widgets?.find((w) => w.name === "frame_id");
        const frameId = frameWidget?.value ?? 0;
        const total = this._ltx_frame_count;
        const resolved =
          frameId < 0
            ? Math.max(0, total + frameId)
            : Math.min(frameId, total - 1);

        const text = `Frame ${resolved}/${total - 1}`;
        ctx.save();
        ctx.font = "12px Arial";
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        const tw = ctx.measureText(text).width;
        ctx.fillRect(this.size[0] - tw - 14, this.size[1] - 24, tw + 10, 20);
        ctx.fillStyle = "#fff";
        ctx.fillText(text, this.size[0] - tw - 9, this.size[1] - 10);
        ctx.restore();
      }
    };
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_TYPE) return;

    // Restore preview when loading a workflow
    const origConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      origConfigure?.apply(this, arguments);
      setTimeout(() => {
        updateFrameCount(this);
        updatePreview(this);
      }, 800);
    };
  },
});
