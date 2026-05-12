-- Tiny 3D demo in Love2D — one file, no deps.
-- Walk around a ground plane with colored cubes.
--
-- Controls:
--   WASD    move
--   Mouse   look
--   Space   jump
--   Esc     quit
--
-- Run with `love .` (Love2D 11.x).

------------------------------------------------------------------------------
-- Tunables
------------------------------------------------------------------------------
local FOV         = math.rad(75)
local NEAR, FAR   = 1, 4000
local MOUSE_SENS  = 0.0025
local MOVE_SPEED  = 220
local EYE_HEIGHT  = 64
local GRAVITY     = 1400
local JUMP_SPEED  = 480

local GROUND_HALF = 1200
local CUBE_COUNT  = 60

------------------------------------------------------------------------------
-- Camera. yaw rotates around +Y, pitch around the camera's right axis.
------------------------------------------------------------------------------
local cam = {
    pos      = {0, EYE_HEIGHT, 400},
    yaw      = 0,
    pitch    = 0,
    velY     = 0,
    onGround = true,
}

-- 4x4 matrix multiply (row-major). Returns a*b.
local function mul(a, b)
    local r = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}}
    for i = 1, 4 do for j = 1, 4 do
        local s = 0
        for k = 1, 4 do s = s + a[i][k] * b[k][j] end
        r[i][j] = s
    end end
    return r
end

-- Build the view-projection matrix. The view is constructed from an explicit
-- camera basis (right / up / forward) so orientation can't end up mis-signed.
local function viewProjection(aspect)
    local cy, sy = math.cos(cam.yaw),   math.sin(cam.yaw)
    local cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)

    -- Forward direction in world space. At yaw=0 pitch=0 this is -Z.
    -- Mouse-right increases yaw, which steers forward toward +X (turns right).
    local fx, fy, fz = sy*cp, sp, -cy*cp
    local rx, rz    = cy, sy            -- right vector lives in the XZ plane
    local ux = -rz*fy                   -- up = right × forward, simplified
    local uy =  rz*fx - rx*fz
    local uz =  rx*fy

    local px, py, pz = cam.pos[1], cam.pos[2], cam.pos[3]
    local view = {
        { rx,   0,  rz, -(rx*px +     rz*pz)},
        { ux,  uy,  uz, -(ux*px + uy*py + uz*pz)},
        {-fx, -fy, -fz,    fx*px + fy*py + fz*pz},
        {  0,   0,   0,  1},
    }

    local f  = 1 / math.tan(FOV * 0.5)
    local nf = 1 / (NEAR - FAR)
    local proj = {
        {f/aspect, 0, 0,                0},
        {0,        f, 0,                0},
        {0,        0, (FAR+NEAR)*nf,    (2*FAR*NEAR)*nf},
        {0,        0, -1,               0},
    }
    return mul(proj, view)
end

------------------------------------------------------------------------------
-- Shader. Vertex pass applies viewProjection; fragment does Lambert + fog.
------------------------------------------------------------------------------
local SKY = {0.55, 0.70, 0.85}

local VERT_SRC = [[
attribute vec3 VertexNormal;
attribute vec3 VertexColor3;
uniform mat4 uViewProj;
uniform vec3 uLight;
varying vec3 vColor;
varying vec3 vWorld;
varying float vShade;
vec4 position(mat4 _t, vec4 vp) {
    float l = max(dot(normalize(VertexNormal), normalize(uLight)), 0.0);
    vShade  = 0.35 + 0.65 * l;
    vColor  = VertexColor3;
    vWorld  = vp.xyz;
    return uViewProj * vp;
}
]]

local FRAG_SRC = [[
varying vec3 vColor;
varying vec3 vWorld;
varying float vShade;
uniform vec3 uSky;
uniform float uFog;
vec4 effect(vec4 _c, Image _t, vec2 _uv, vec2 _sc) {
    float fog = clamp(length(vWorld.xz) / uFog, 0.0, 1.0);
    return vec4(mix(vColor * vShade, uSky, fog), 1.0);
}
]]

local VFMT = {
    {"VertexPosition", "float", 3},
    {"VertexNormal",   "float", 3},
    {"VertexColor3",   "float", 3},
}

------------------------------------------------------------------------------
-- Mesh helpers. pushQuad emits two triangles wound so the visible side from
-- +normal faces outward (CCW), letting backface culling stay on.
------------------------------------------------------------------------------
local function pushQuad(V, I, a, b, c, d, n, col)
    local base = #V
    local function v(p) return {p[1], p[2], p[3], n[1], n[2], n[3], col[1], col[2], col[3]} end
    V[#V+1] = v(a); V[#V+1] = v(b); V[#V+1] = v(c); V[#V+1] = v(d)
    -- Triangles (A, C, B) and (A, D, C). For each face below, the cross
    -- (C-A) × (B-A) points along the named normal — verified by hand.
    I[#I+1] = base+1; I[#I+1] = base+3; I[#I+1] = base+2
    I[#I+1] = base+1; I[#I+1] = base+4; I[#I+1] = base+3
end

local function pushCube(V, I, cx, cy, cz, s, col)
    local h = s * 0.5
    local x0, y0, z0 = cx-h, cy-h, cz-h
    local x1, y1, z1 = cx+h, cy+h, cz+h
    pushQuad(V, I, {x0,y1,z1},{x1,y1,z1},{x1,y0,z1},{x0,y0,z1}, { 0, 0, 1}, col)
    pushQuad(V, I, {x1,y1,z0},{x0,y1,z0},{x0,y0,z0},{x1,y0,z0}, { 0, 0,-1}, col)
    pushQuad(V, I, {x0,y1,z0},{x0,y1,z1},{x0,y0,z1},{x0,y0,z0}, {-1, 0, 0}, col)
    pushQuad(V, I, {x1,y1,z1},{x1,y1,z0},{x1,y0,z0},{x1,y0,z1}, { 1, 0, 0}, col)
    pushQuad(V, I, {x0,y1,z0},{x1,y1,z0},{x1,y1,z1},{x0,y1,z1}, { 0, 1, 0}, col)
    pushQuad(V, I, {x0,y0,z1},{x1,y0,z1},{x1,y0,z0},{x0,y0,z0}, { 0,-1, 0}, col)
end

local function buildWorld()
    local V, I = {}, {}
    pushQuad(V, I,
        {-GROUND_HALF, 0, -GROUND_HALF},
        { GROUND_HALF, 0, -GROUND_HALF},
        { GROUND_HALF, 0,  GROUND_HALF},
        {-GROUND_HALF, 0,  GROUND_HALF},
        {0, 1, 0}, {0.22, 0.28, 0.32})

    math.randomseed(1)
    for _ = 1, CUBE_COUNT do
        local x = (math.random() * 2 - 1) * (GROUND_HALF - 100)
        local z = (math.random() * 2 - 1) * (GROUND_HALF - 100)
        local h = 30 + math.random() * 80
        local col = {0.4 + math.random()*0.55, 0.4 + math.random()*0.55, 0.4 + math.random()*0.55}
        pushCube(V, I, x, h * 0.5, z, h, col)
    end

    local mesh = love.graphics.newMesh(VFMT, V, "triangles", "static")
    mesh:setVertexMap(I)
    return mesh
end

------------------------------------------------------------------------------
-- Love callbacks
------------------------------------------------------------------------------
local world, shader

function love.load()
    love.window.setTitle("Love2D 3D — single-file demo")
    -- Ask for a depth buffer on the main framebuffer so we can z-test
    -- directly against the screen, no intermediate canvas required.
    love.window.setMode(1280, 720, { resizable = true, vsync = true, depth = 16 })
    love.mouse.setRelativeMode(true)

    shader = love.graphics.newShader(FRAG_SRC, VERT_SRC)
    world  = buildWorld()
end

function love.mousemoved(_, _, dx, dy)
    -- Raw deltas — never multiply by dt; they're already time-independent.
    cam.yaw   = cam.yaw   + dx * MOUSE_SENS
    cam.pitch = cam.pitch - dy * MOUSE_SENS
    local lim = math.pi * 0.5 - 0.01
    if cam.pitch >  lim then cam.pitch =  lim end
    if cam.pitch < -lim then cam.pitch = -lim end
end

function love.keypressed(key)
    if key == "escape" then love.event.quit(0) end
    if key == "space" and cam.onGround then
        cam.velY = JUMP_SPEED
        cam.onGround = false
    end
end

function love.update(dt)
    -- Build horizontal velocity from input + camera yaw.
    local kb = love.keyboard
    local fwd  = (kb.isDown("w") and 1 or 0) - (kb.isDown("s") and 1 or 0)
    local strafe = (kb.isDown("d") and 1 or 0) - (kb.isDown("a") and 1 or 0)
    local sy, cy = math.sin(cam.yaw), math.cos(cam.yaw)
    local vx = sy * fwd + cy * strafe
    local vz = -cy * fwd + sy * strafe
    local len = math.sqrt(vx*vx + vz*vz)
    if len > 0 then vx, vz = vx / len, vz / len end

    cam.pos[1] = cam.pos[1] + vx * MOVE_SPEED * dt
    cam.pos[3] = cam.pos[3] + vz * MOVE_SPEED * dt

    -- Gravity + ground clamp at eye-height.
    cam.velY  = cam.velY - GRAVITY * dt
    cam.pos[2] = cam.pos[2] + cam.velY * dt
    if cam.pos[2] <= EYE_HEIGHT then
        cam.pos[2]  = EYE_HEIGHT
        cam.velY    = 0
        cam.onGround = true
    end
end

function love.draw()
    local w, h = love.graphics.getDimensions()
    local vp   = viewProjection(w / h)

    love.graphics.clear(SKY[1], SKY[2], SKY[3], 1, true, true)
    love.graphics.setDepthMode("lequal", true)
    love.graphics.setMeshCullMode("back")
    love.graphics.setShader(shader)
    shader:send("uViewProj", "row", vp)
    shader:send("uLight",    {0.4, 1.0, 0.3})
    shader:send("uSky",      SKY)
    shader:send("uFog",      GROUND_HALF * 1.4)
    love.graphics.draw(world)

    love.graphics.setShader()
    love.graphics.setDepthMode()
    love.graphics.setMeshCullMode("none")
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.print("WASD move, mouse look, space jump, esc quit", 8, 8)
end
