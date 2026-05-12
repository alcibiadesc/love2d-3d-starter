-- Tiny 3D demo in Love2D — one file, no deps.
-- Flat ground plane, WASD to move, mouse to look. That's it.
--
-- The whole thing is: a hand-rolled row-major mat4, a yaw/pitch camera,
-- and one shader. Love2D's 2D API can render 3D if you set the depth
-- buffer up yourself and ship a viewProjection matrix to a custom shader.
--
-- Controls:
--   WASD           move
--   Mouse          look
--   Space / LCtrl  fly up / down
--   Esc            quit
--
-- Run with `love .` (Love2D 11.x).

------------------------------------------------------------------------------
-- Tunables
------------------------------------------------------------------------------
local FOV        = math.rad(75)
local NEAR, FAR  = 1, 4000
local MOUSE_SENS = 0.0025
local MOVE_SPEED = 200      -- world units / second

local PLANE_HALF = 1000     -- ground extends this far in each direction
local GRID_CELL  = 50       -- shader grid line every N world units

------------------------------------------------------------------------------
-- mat4 — row-major. Sent to shaders with :send(name, "row", m).
------------------------------------------------------------------------------
local mat4 = {}

function mat4.translation(x, y, z)
    return {{1,0,0,x},{0,1,0,y},{0,0,1,z},{0,0,0,1}}
end

function mat4.rotationX(a)
    local c, s = math.cos(a), math.sin(a)
    return {{1,0,0,0},{0,c,-s,0},{0,s,c,0},{0,0,0,1}}
end

function mat4.rotationY(a)
    local c, s = math.cos(a), math.sin(a)
    return {{c,0,s,0},{0,1,0,0},{-s,0,c,0},{0,0,0,1}}
end

function mat4.perspective(fovY, aspect, near, far)
    local f  = 1 / math.tan(fovY * 0.5)
    local nf = 1 / (near - far)
    return {
        {f/aspect, 0, 0,             0},
        {0,        f, 0,             0},
        {0,        0, (far+near)*nf, (2*far*near)*nf},
        {0,        0, -1,            0},
    }
end

function mat4.mul(a, b)
    local r = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}}
    for i = 1, 4 do
        for j = 1, 4 do
            local s = 0
            for k = 1, 4 do s = s + a[i][k] * b[k][j] end
            r[i][j] = s
        end
    end
    return r
end

------------------------------------------------------------------------------
-- Camera. Yaw around Y, pitch around X (clamped to avoid gimbal flip).
------------------------------------------------------------------------------
local camera = {
    pos   = {0, 60, 200},
    yaw   = 0,
    pitch = 0,
}

function camera:rotate(dx, dy)
    self.yaw   = self.yaw   + dx * MOUSE_SENS
    self.pitch = self.pitch - dy * MOUSE_SENS
    local lim  = math.pi * 0.5 - 0.01
    if self.pitch >  lim then self.pitch =  lim end
    if self.pitch < -lim then self.pitch = -lim end
end

function camera:forwardXZ()
    return -math.sin(self.yaw), -math.cos(self.yaw)
end

function camera:rightXZ()
    return math.cos(self.yaw), -math.sin(self.yaw)
end

function camera:viewProjection(aspect)
    local p  = mat4.perspective(FOV, aspect, NEAR, FAR)
    local ry = mat4.rotationY(-self.yaw)
    local rx = mat4.rotationX(-self.pitch)
    local t  = mat4.translation(-self.pos[1], -self.pos[2], -self.pos[3])
    return mat4.mul(p, mat4.mul(rx, mat4.mul(ry, t)))
end

------------------------------------------------------------------------------
-- Shader. Vertex pass = viewProjection. Fragment paints a grid based on
-- world-space XZ and fades to the sky color with distance fog.
------------------------------------------------------------------------------
local VERT_SRC = [[
uniform mat4 uViewProj;
varying vec3 vWorld;

vec4 position(mat4 _t, vec4 vp) {
    vWorld = vp.xyz;
    return uViewProj * vp;
}
]]

local FRAG_SRC = [[
varying vec3 vWorld;
uniform float uGridCell;
uniform float uFog;

vec4 effect(vec4 _c, Image _t, vec2 _uv, vec2 _sc) {
    vec2 g  = abs(fract(vWorld.xz / uGridCell - 0.5) - 0.5) * uGridCell;
    float d = min(g.x, g.y);
    float w = fwidth(d) * 1.2;
    float line = 1.0 - smoothstep(0.0, w, d);

    vec3 floorCol = vec3(0.18, 0.20, 0.24);
    vec3 lineCol  = vec3(0.55, 0.62, 0.72);
    vec3 col      = mix(floorCol, lineCol, line);

    float fog = clamp(length(vWorld.xz) / uFog, 0.0, 1.0);
    vec3  sky = vec3(0.55, 0.70, 0.85);
    return vec4(mix(col, sky, fog), 1.0);
}
]]

------------------------------------------------------------------------------
-- One quad covering the ground plane. Position only — shader does the rest.
------------------------------------------------------------------------------
local VERTEX_FORMAT = {{"VertexPosition", "float", 3}}

local function buildPlaneMesh()
    local h = PLANE_HALF
    local verts = {
        {-h, 0, -h},
        { h, 0, -h},
        { h, 0,  h},
        {-h, 0,  h},
    }
    -- CCW from above (+Y normal) so backface culling can stay enabled.
    local indices = {1, 3, 2,  1, 4, 3}
    local mesh = love.graphics.newMesh(VERTEX_FORMAT, verts, "triangles", "static")
    mesh:setVertexMap(indices)
    return mesh
end

------------------------------------------------------------------------------
-- Love callbacks
------------------------------------------------------------------------------
local plane, shader, canvas

local function newCanvas(w, h)
    local c = love.graphics.newCanvas(w, h, { format = "rgba8" })
    c:setFilter("nearest", "nearest")
    return c
end

function love.load()
    love.window.setTitle("Love2D 3D — single-file demo")
    love.window.setMode(1280, 720, { resizable = true, vsync = 1 })
    love.mouse.setRelativeMode(true)

    shader = love.graphics.newShader(FRAG_SRC, VERT_SRC)
    plane  = buildPlaneMesh()
    canvas = newCanvas(love.graphics.getDimensions())
end

function love.resize(w, h)
    canvas = newCanvas(w, h)
end

function love.mousemoved(_, _, dx, dy)
    -- Raw deltas — do NOT multiply by dt, they're already time-independent.
    camera:rotate(dx, dy)
end

function love.keypressed(key)
    if key == "escape" then love.event.quit(0) end
end

function love.update(dt)
    local fx, fz = camera:forwardXZ()
    local rx, rz = camera:rightXZ()

    local kb = love.keyboard
    local forward = (kb.isDown("w") and 1 or 0) - (kb.isDown("s") and 1 or 0)
    local side    = (kb.isDown("d") and 1 or 0) - (kb.isDown("a") and 1 or 0)
    local up      = (kb.isDown("space") and 1 or 0) - (kb.isDown("lctrl","lshift") and 1 or 0)

    local vx = fx * forward + rx * side
    local vz = fz * forward + rz * side
    local len = math.sqrt(vx*vx + vz*vz)
    if len > 0 then vx, vz = vx / len, vz / len end

    camera.pos[1] = camera.pos[1] + vx * MOVE_SPEED * dt
    camera.pos[3] = camera.pos[3] + vz * MOVE_SPEED * dt
    camera.pos[2] = camera.pos[2] + up * MOVE_SPEED * dt
end

function love.draw()
    local w, h   = love.graphics.getDimensions()
    local aspect = w / h
    local vp     = camera:viewProjection(aspect)

    -- 3D pass: depth test on, render into a canvas with depth attached.
    love.graphics.setCanvas({ canvas, depth = true })
    love.graphics.clear(0.55, 0.70, 0.85, 1, true, true)  -- sky + depth
    love.graphics.setDepthMode("lequal", true)
    love.graphics.setMeshCullMode("back")
    love.graphics.setShader(shader)
    shader:send("uViewProj", "row", vp)
    shader:send("uGridCell", GRID_CELL)
    shader:send("uFog",      PLANE_HALF * 0.9)
    love.graphics.draw(plane)

    -- Reset to 2D.
    love.graphics.setShader()
    love.graphics.setDepthMode()
    love.graphics.setMeshCullMode("none")
    love.graphics.setCanvas()

    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.draw(canvas, 0, 0)

    -- HUD.
    love.graphics.print(
        ("FPS %d   pos %.0f, %.0f, %.0f"):format(
            love.timer.getFPS(), camera.pos[1], camera.pos[2], camera.pos[3]),
        8, 8)
    love.graphics.print("WASD move, mouse look, space/ctrl fly, esc quit", 8, 24)
end
