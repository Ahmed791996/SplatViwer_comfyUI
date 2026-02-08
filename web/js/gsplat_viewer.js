import { app } from "../../scripts/app.js";

// ─── Shader sources ────────────────────────────────────────────

const POINT_VS = `#version 300 es
uniform mat4 u_mvp;
uniform float u_pointSize;
in vec3 a_pos;
in vec3 a_col;
out vec3 v_col;
void main(){
    v_col = a_col;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    gl_PointSize = u_pointSize;
}`;

const POINT_FS = `#version 300 es
precision mediump float;
in vec3 v_col;
out vec4 fragColor;
void main(){
    vec2 c = gl_PointCoord - 0.5;
    if(dot(c,c)>0.25) discard;
    fragColor = vec4(v_col, 1.0);
}`;

const SPLAT_VS = `#version 300 es
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec2 u_viewport;

in vec3 a_center;   // splat world position
in vec3 a_col;
in float a_opacity;
in vec3 a_scale;
in vec4 a_rot;      // quaternion wxyz
in vec2 a_quad;     // per-vertex quad corner (-1 or +1)

out vec3 v_col;
out float v_opacity;
out vec2 v_uv;

mat3 quatToMat(vec4 q){
    float w=q.x, x=q.y, y=q.z, z=q.w;
    return mat3(
        1.0-2.0*(y*y+z*z), 2.0*(x*y+w*z),     2.0*(x*z-w*y),
        2.0*(x*y-w*z),     1.0-2.0*(x*x+z*z),  2.0*(y*z+w*x),
        2.0*(x*z+w*y),     2.0*(y*z-w*x),       1.0-2.0*(x*x+y*y)
    );
}

void main(){
    v_col = a_col;
    v_opacity = a_opacity;
    v_uv = a_quad;

    // Transform center to view space
    vec4 viewCenter = u_view * vec4(a_center, 1.0);

    // Build rotation matrix and scale in view space
    mat3 R = mat3(u_view) * quatToMat(a_rot);
    vec3 scaled = R * (a_scale * vec3(a_quad, 0.0));

    // Offset in view space — scale by 2 for visual size
    vec4 viewPos = viewCenter + vec4(scaled * 2.0, 0.0);

    gl_Position = u_proj * viewPos;
}`;

const SPLAT_FS = `#version 300 es
precision mediump float;
in vec3 v_col;
in float v_opacity;
in vec2 v_uv;
out vec4 fragColor;
void main(){
    float g = exp(-0.5 * dot(v_uv, v_uv));
    float alpha = v_opacity * g;
    if(alpha < 1.0/255.0) discard;
    fragColor = vec4(v_col * alpha, alpha);
}`;

// ─── GL helpers ────────────────────────────────────────────────

function createShader(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.error("Shader compile error:", gl.getShaderInfoLog(s));
        gl.deleteShader(s);
        return null;
    }
    return s;
}

function createProgram(gl, vsSrc, fsSrc) {
    const vs = createShader(gl, gl.VERTEX_SHADER, vsSrc);
    const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSrc);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        console.error("Program link error:", gl.getProgramInfoLog(prog));
        return null;
    }
    return prog;
}

// ─── Matrix math (minimal) ─────────────────────────────────────

function mat4Perspective(fovY, aspect, near, far) {
    const f = 1.0 / Math.tan(fovY / 2);
    const nf = 1 / (near - far);
    return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, 2 * far * near * nf, 0,
    ]);
}

function mat4LookAt(eye, center, up) {
    const zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
    let len = 1 / Math.hypot(zx, zy, zz);
    const z = [zx * len, zy * len, zz * len];
    const xx = up[1] * z[2] - up[2] * z[1];
    const xy = up[2] * z[0] - up[0] * z[2];
    const xz = up[0] * z[1] - up[1] * z[0];
    len = 1 / Math.hypot(xx, xy, xz);
    const x = [xx * len, xy * len, xz * len];
    const y = [x[1] * z[2] - x[2] * z[1], x[2] * z[0] - x[0] * z[2], x[0] * z[1] - x[1] * z[0]];
    return new Float32Array([
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]),
        -(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]),
        -(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]),
        1,
    ]);
}

function mat4Multiply(a, b) {
    const o = new Float32Array(16);
    for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++) {
            o[j * 4 + i] = a[i] * b[j * 4] + a[4 + i] * b[j * 4 + 1] + a[8 + i] * b[j * 4 + 2] + a[12 + i] * b[j * 4 + 3];
        }
    return o;
}

// ─── ComfyUI Extension ────────────────────────────────────────

app.registerExtension({
    name: "GSplatViewer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "GaussianSplatViewer") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);
            this._gsplatState = {
                mode: "splat", // "points" or "splat"
                canvas: null,
                gl: null,
                data: null,
                camera: { theta: 0.45, phi: 0.6, dist: 5.0, target: [0, 0, 0] },
                dragging: false,
                lastMouse: [0, 0],
                rightDrag: false,
                pointProg: null,
                splatProg: null,
                animId: null,
            };
            this.size = [520, 580];
        };

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            origOnExecuted?.apply(this, arguments);
            if (!output?.splat_data?.[0]) return;
            const parsed = JSON.parse(output.splat_data[0]);
            if (parsed.error) {
                console.error("GSplatViewer:", parsed.error);
                return;
            }
            this._gsplatState.data = parsed;
            this._initViewer();
        };

        nodeType.prototype._initViewer = function () {
            const st = this._gsplatState;
            if (!st.data) return;

            // Create canvas if needed
            if (!st.canvas) {
                const canvas = document.createElement("canvas");
                canvas.width = 512;
                canvas.height = 512;
                canvas.style.cssText = "border:1px solid #555;border-radius:4px;cursor:grab;";

                // Mouse controls
                canvas.addEventListener("mousedown", (e) => {
                    e.preventDefault();
                    st.dragging = true;
                    st.rightDrag = e.button === 2;
                    st.lastMouse = [e.clientX, e.clientY];
                    canvas.style.cursor = "grabbing";
                });
                canvas.addEventListener("contextmenu", (e) => e.preventDefault());
                window.addEventListener("mouseup", () => {
                    st.dragging = false;
                    st.rightDrag = false;
                    canvas.style.cursor = "grab";
                });
                window.addEventListener("mousemove", (e) => {
                    if (!st.dragging) return;
                    const dx = e.clientX - st.lastMouse[0];
                    const dy = e.clientY - st.lastMouse[1];
                    st.lastMouse = [e.clientX, e.clientY];
                    if (st.rightDrag) {
                        // Pan
                        const panSpeed = st.camera.dist * 0.002;
                        const ct = Math.cos(st.camera.theta), st2 = Math.sin(st.camera.theta);
                        st.camera.target[0] -= (dx * ct) * panSpeed;
                        st.camera.target[2] -= (-dx * st2) * panSpeed;
                        st.camera.target[1] += dy * panSpeed;
                    } else {
                        // Orbit
                        st.camera.theta -= dx * 0.005;
                        st.camera.phi = Math.max(0.05, Math.min(Math.PI - 0.05, st.camera.phi - dy * 0.005));
                    }
                });
                canvas.addEventListener("wheel", (e) => {
                    e.preventDefault();
                    st.camera.dist = Math.max(0.5, st.camera.dist * (1 + e.deltaY * 0.001));
                }, { passive: false });

                st.canvas = canvas;

                const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
                if (!gl) { console.error("WebGL2 not supported"); return; }
                st.gl = gl;
                st.pointProg = createProgram(gl, POINT_VS, POINT_FS);
                st.splatProg = createProgram(gl, SPLAT_VS, SPLAT_FS);
            }

            this._uploadBuffers();
            this._startRender();
        };

        nodeType.prototype._uploadBuffers = function () {
            const st = this._gsplatState;
            const gl = st.gl;
            const d = st.data;
            const N = d.count;

            // Flatten arrays
            const pos = new Float32Array(N * 3);
            const col = new Float32Array(N * 3);
            const scl = new Float32Array(N * 3);
            const rot = new Float32Array(N * 4);
            const opa = new Float32Array(N);

            for (let i = 0; i < N; i++) {
                pos[i * 3] = d.positions[i][0];
                pos[i * 3 + 1] = d.positions[i][1];
                pos[i * 3 + 2] = d.positions[i][2];
                col[i * 3] = d.colors[i][0];
                col[i * 3 + 1] = d.colors[i][1];
                col[i * 3 + 2] = d.colors[i][2];
                scl[i * 3] = d.scales[i][0];
                scl[i * 3 + 1] = d.scales[i][1];
                scl[i * 3 + 2] = d.scales[i][2];
                rot[i * 4] = d.rotations[i][0];
                rot[i * 4 + 1] = d.rotations[i][1];
                rot[i * 4 + 2] = d.rotations[i][2];
                rot[i * 4 + 3] = d.rotations[i][3];
                opa[i] = d.opacities[i];
            }

            // Auto-center camera on data
            let cx = 0, cy = 0, cz = 0;
            for (let i = 0; i < N; i++) {
                cx += pos[i * 3]; cy += pos[i * 3 + 1]; cz += pos[i * 3 + 2];
            }
            st.camera.target = [cx / N, cy / N, cz / N];

            // Estimate distance
            let maxR = 0;
            for (let i = 0; i < N; i++) {
                const dx = pos[i*3] - st.camera.target[0];
                const dy = pos[i*3+1] - st.camera.target[1];
                const dz = pos[i*3+2] - st.camera.target[2];
                maxR = Math.max(maxR, Math.sqrt(dx*dx+dy*dy+dz*dz));
            }
            st.camera.dist = maxR * 1.5 || 5.0;

            function makeBuf(arr) {
                const b = gl.createBuffer();
                gl.bindBuffer(gl.ARRAY_BUFFER, b);
                gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
                return b;
            }

            st.buffers = {
                pos: makeBuf(pos),
                col: makeBuf(col),
                scl: makeBuf(scl),
                rot: makeBuf(rot),
                opa: makeBuf(opa),
                count: N,
                rawPos: pos,
            };

            // Quad buffer for splat instancing
            const quad = new Float32Array([-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1]);
            st.buffers.quad = makeBuf(quad);
        };

        nodeType.prototype._startRender = function () {
            const st = this._gsplatState;
            if (st.animId) cancelAnimationFrame(st.animId);

            const render = () => {
                st.animId = requestAnimationFrame(render);
                const gl = st.gl;
                const cam = st.camera;
                const w = st.canvas.width, h = st.canvas.height;

                gl.viewport(0, 0, w, h);
                gl.clearColor(0.12, 0.12, 0.15, 1.0);
                gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

                // Camera
                const eye = [
                    cam.target[0] + cam.dist * Math.sin(cam.phi) * Math.cos(cam.theta),
                    cam.target[1] + cam.dist * Math.cos(cam.phi),
                    cam.target[2] + cam.dist * Math.sin(cam.phi) * Math.sin(cam.theta),
                ];
                const view = mat4LookAt(eye, cam.target, [0, 1, 0]);
                const proj = mat4Perspective(Math.PI / 4, w / h, 0.1, 1000.0);
                const mvp = mat4Multiply(proj, view);

                if (st.mode === "points") {
                    this._drawPoints(gl, mvp);
                } else {
                    this._drawSplats(gl, view, proj);
                }
            };
            render();
        };

        nodeType.prototype._drawPoints = function (gl, mvp) {
            const st = this._gsplatState;
            const prog = st.pointProg;
            gl.useProgram(prog);
            gl.enable(gl.DEPTH_TEST);
            gl.disable(gl.BLEND);

            // Get point size from widget
            const pointSize = st.data?.point_size || 3.0;
            gl.uniformMatrix4fv(gl.getUniformLocation(prog, "u_mvp"), false, mvp);
            gl.uniform1f(gl.getUniformLocation(prog, "u_pointSize"), pointSize);

            // Position
            const aPos = gl.getAttribLocation(prog, "a_pos");
            gl.bindBuffer(gl.ARRAY_BUFFER, st.buffers.pos);
            gl.enableVertexAttribArray(aPos);
            gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);

            // Color
            const aCol = gl.getAttribLocation(prog, "a_col");
            gl.bindBuffer(gl.ARRAY_BUFFER, st.buffers.col);
            gl.enableVertexAttribArray(aCol);
            gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, 0, 0);

            gl.drawArrays(gl.POINTS, 0, st.buffers.count);
        };

        nodeType.prototype._drawSplats = function (gl, view, proj) {
            const st = this._gsplatState;
            const prog = st.splatProg;
            const N = st.buffers.count;
            gl.useProgram(prog);

            gl.disable(gl.DEPTH_TEST);
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

            gl.uniformMatrix4fv(gl.getUniformLocation(prog, "u_view"), false, view);
            gl.uniformMatrix4fv(gl.getUniformLocation(prog, "u_proj"), false, proj);
            gl.uniform2f(gl.getUniformLocation(prog, "u_viewport"), st.canvas.width, st.canvas.height);

            // Sort splats back-to-front
            const rawPos = st.buffers.rawPos;
            const indices = new Uint32Array(N);
            const depths = new Float32Array(N);
            for (let i = 0; i < N; i++) {
                indices[i] = i;
                // Depth in view space: dot with view z-axis (row 2 of view matrix)
                depths[i] = view[2] * rawPos[i*3] + view[6] * rawPos[i*3+1] + view[10] * rawPos[i*3+2] + view[14];
            }
            indices.sort((a, b) => depths[a] - depths[b]);

            // Build sorted instanced arrays
            const sortedCenter = new Float32Array(N * 3);
            const sortedCol = new Float32Array(N * 3);
            const sortedOpa = new Float32Array(N);
            const sortedScl = new Float32Array(N * 3);
            const sortedRot = new Float32Array(N * 4);
            const d = st.data;
            for (let k = 0; k < N; k++) {
                const i = indices[k];
                sortedCenter[k*3]   = d.positions[i][0];
                sortedCenter[k*3+1] = d.positions[i][1];
                sortedCenter[k*3+2] = d.positions[i][2];
                sortedCol[k*3]   = d.colors[i][0];
                sortedCol[k*3+1] = d.colors[i][1];
                sortedCol[k*3+2] = d.colors[i][2];
                sortedOpa[k] = d.opacities[i];
                sortedScl[k*3]   = d.scales[i][0];
                sortedScl[k*3+1] = d.scales[i][1];
                sortedScl[k*3+2] = d.scales[i][2];
                sortedRot[k*4]   = d.rotations[i][0];
                sortedRot[k*4+1] = d.rotations[i][1];
                sortedRot[k*4+2] = d.rotations[i][2];
                sortedRot[k*4+3] = d.rotations[i][3];
            }

            // Upload sorted data
            function updateBuf(buf, data) {
                gl.bindBuffer(gl.ARRAY_BUFFER, buf);
                gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
            }

            const centerBuf = st.buffers._sortedCenter || gl.createBuffer();
            const colBuf2   = st.buffers._sortedCol    || gl.createBuffer();
            const opaBuf2   = st.buffers._sortedOpa    || gl.createBuffer();
            const sclBuf2   = st.buffers._sortedScl    || gl.createBuffer();
            const rotBuf2   = st.buffers._sortedRot    || gl.createBuffer();
            st.buffers._sortedCenter = centerBuf;
            st.buffers._sortedCol = colBuf2;
            st.buffers._sortedOpa = opaBuf2;
            st.buffers._sortedScl = sclBuf2;
            st.buffers._sortedRot = rotBuf2;

            updateBuf(centerBuf, sortedCenter);
            updateBuf(colBuf2, sortedCol);
            updateBuf(opaBuf2, sortedOpa);
            updateBuf(sclBuf2, sortedScl);
            updateBuf(rotBuf2, sortedRot);

            // Setup instanced attributes
            const ext = gl;

            function setAttr(name, buf, size, divisor) {
                const loc = gl.getAttribLocation(prog, name);
                if (loc < 0) return;
                gl.bindBuffer(gl.ARRAY_BUFFER, buf);
                gl.enableVertexAttribArray(loc);
                gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
                ext.vertexAttribDivisor(loc, divisor);
            }

            // Quad (per-vertex, divisor=0)
            setAttr("a_quad", st.buffers.quad, 2, 0);
            // Per-instance (divisor=1)
            setAttr("a_center", centerBuf, 3, 1);
            setAttr("a_col", colBuf2, 3, 1);
            setAttr("a_opacity", opaBuf2, 1, 1);
            setAttr("a_scale", sclBuf2, 3, 1);
            setAttr("a_rot", rotBuf2, 4, 1);

            ext.drawArraysInstanced(gl.TRIANGLES, 0, 6, N);

            // Clean up divisors
            for (const n of ["a_center", "a_col", "a_opacity", "a_scale", "a_rot"]) {
                const loc = gl.getAttribLocation(prog, n);
                if (loc >= 0) ext.vertexAttribDivisor(loc, 0);
            }
        };

        // Custom drawing inside ComfyUI node
        const origDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function (ctx) {
            origDrawBackground?.apply(this, arguments);
            const st = this._gsplatState;
            if (!st.canvas) return;

            // Draw toggle button
            ctx.fillStyle = "#333";
            ctx.fillRect(10, 30, 100, 24);
            ctx.fillStyle = "#fff";
            ctx.font = "12px sans-serif";
            ctx.fillText(st.mode === "points" ? "Mode: Points" : "Mode: Splats", 18, 46);

            // Draw capture button
            ctx.fillStyle = "#2a5";
            ctx.fillRect(120, 30, 90, 24);
            ctx.fillStyle = "#fff";
            ctx.fillText("Capture", 140, 46);

            // Draw WebGL canvas as image
            ctx.drawImage(st.canvas, 4, 60, this.size[0] - 8, this.size[1] - 70);
        };

        const origOnMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e, localPos) {
            const st = this._gsplatState;
            // Toggle button
            if (localPos[0] > 10 && localPos[0] < 110 && localPos[1] > 30 && localPos[1] < 54) {
                st.mode = st.mode === "points" ? "splat" : "points";
                return true;
            }
            // Capture button
            if (localPos[0] > 120 && localPos[0] < 210 && localPos[1] > 30 && localPos[1] < 54) {
                this._captureImage();
                return true;
            }
            // Forward mouse to canvas viewport area
            if (localPos[1] > 60) {
                st.dragging = true;
                st.rightDrag = e.button === 2;
                st.lastMouse = [e.canvasX, e.canvasY];
                return true;
            }
            return origOnMouseDown?.apply(this, arguments);
        };

        nodeType.prototype._captureImage = function () {
            const st = this._gsplatState;
            if (!st.canvas) return;

            // Convert canvas to base64 PNG
            const dataURL = st.canvas.toDataURL("image/png");

            // Find or create the hidden widget for captured image data
            let widget = this.widgets?.find(w => w.name === "captured_image_data");
            if (!widget) {
                widget = this.addWidget("text", "captured_image_data", dataURL, () => {}, {
                    serialize: true,
                });
                widget.hidden = true;
            }
            widget.value = dataURL;

            // Queue the node for re-execution to process the captured image
            if (app.queuePrompt) {
                console.log("Image captured, queuing prompt...");
                app.queuePrompt(0, 1);
            }
        };

        const origOnMouseMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function (e, localPos) {
            const st = this._gsplatState;
            if (st.dragging && localPos[1] > 60) {
                const dx = e.canvasX - st.lastMouse[0];
                const dy = e.canvasY - st.lastMouse[1];
                st.lastMouse = [e.canvasX, e.canvasY];
                if (st.rightDrag) {
                    const panSpeed = st.camera.dist * 0.002;
                    const ct = Math.cos(st.camera.theta), sn = Math.sin(st.camera.theta);
                    st.camera.target[0] -= dx * ct * panSpeed;
                    st.camera.target[2] += dx * sn * panSpeed;
                    st.camera.target[1] += dy * panSpeed;
                } else {
                    st.camera.theta -= dx * 0.005;
                    st.camera.phi = Math.max(0.05, Math.min(Math.PI - 0.05, st.camera.phi - dy * 0.005));
                }
                return true;
            }
            return origOnMouseMove?.apply(this, arguments);
        };

        const origOnMouseUp = nodeType.prototype.onMouseUp;
        nodeType.prototype.onMouseUp = function (e) {
            this._gsplatState.dragging = false;
            this._gsplatState.rightDrag = false;
            return origOnMouseUp?.apply(this, arguments);
        };

        // Scroll to zoom inside node
        const origOnMouseWheel = nodeType.prototype.onMouseWheel;
        nodeType.prototype.onMouseWheel = function (e, localPos) {
            if (localPos && localPos[1] > 60) {
                const st = this._gsplatState;
                st.camera.dist = Math.max(0.5, st.camera.dist * (1 + e.deltaY * 0.001));
                return true;
            }
            return origOnMouseWheel?.apply(this, arguments);
        };
    },
});
