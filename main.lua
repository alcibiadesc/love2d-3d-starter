-- Tiny 3D demo in Love2D — one file, no deps.
-- Flat ground with colored cubes. Walk around, jump, look with the mouse.
--
-- The whole thing is: a hand-rolled row-major mat4, a yaw/pitch camera,
-- and a tiny shader. Love2D's 2D API renders 3D fine once the main
-- framebuffer has a depth buffer (asked for in setMode) and you ship a
-- viewProjection matrix to a custom shader.
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
local PLAYER_R    = 18

local GROUND_HALF = 1200
local CUBE_SIZE   = 40
local CUBE_COUNT  = 64

------------------------------------------------------------------------------
-- mat4 — row-major. Send to shaders with :send(name, "row", m).
------------------------------------------------------------------------------
local mat4 = {}

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

-- Standard GL perspective. Right-handed: camera looks down -Z in view space.
-- Note the -1 in the bottom row: it makes the perspective divide use -z_view
-- as w, so points in front (z_view < 0) end up with positive w.
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

------------------------------------------------------------------------------
-- Camera. yaw rotates around +Y; pitch rotates around the camera's right
-- axis. Conventions chosen so mouse-right turns the view right (forward
-- moves toward +X) and mouse-up tilts the view up.
------------------------------------------------------------------------------
local camera = {
    pos      = {0, EYE_HEIGHT, 400},
    yaw      = 0,
    pitch    = 0,
    velY     = 0,
    onGround = true,
}

function camera:rotate(dx, dy)
    self.yaw   = self.yaw   + dx * MOUSE_SENS
    self.pitch = self.pitch - dy * MOUSE_SENS
    local lim  = math.pi * 0.5 - 0.01
    if self.pitch >  lim then self.pitch =  lim end
    if self.pitch < -lim then self.pitch = -lim end
end

-- Forward vector projected onto the XZ plane (used for WASD).
-- At yaw=0 returns (0, -1) so W moves toward -Z.
function camera:forwardXZ()
    return math.sin(self.yaw), -math.cos(self.yaw)
end

-- Right vector projected onto XZ (used for strafe).
-- At yaw=0 returns (1, 0) so D moves toward +X.
function camera:rightXZ()
    return math.cos(self.yaw), math.sin(self.yaw)
end

-- Build the view-projection by composing an explicit world-space camera
-- basis (right / up / forward) into a look-at view matrix, then the
-- perspective. The explicit basis makes orientation impossible to mis-sign.
function camera:viewProjection(aspect)
    local cy, sy = math.cos(self.yaw),   math.sin(self.yaw)
    local cp, sp = math.cos(self.pitch), math.sin(self.pitch)

    -- World-space camera basis. Forward at yaw=0 pitch=0 is -Z.
    local fx, fy, fz = sy*cp, sp, -cy*cp
    local rx, ry, rz = cy,    0,  sy
    -- up = right × forward (verified: at yaw=0 pitch=0 yields +Y).
    local ux = ry*fz - rz*fy
    local uy = rz*fx - rx*fz
    local uz = rx*fy - ry*fx

    local px, py, pz = self.pos[1], self.pos[2], self.pos[3]

    -- View matrix: rows are right, up, -forward; translate by -pos.
    -- (Each row dotted with a world point gives that point's view-space coord.)
    local view = {
        { rx,  ry,  rz, -(rx*px + ry*py + rz*pz)},
        { ux,  uy,  uz, -(ux*px + uy*py + uz*pz)},
        {-fx, -fy, -fz, -(-fx*px + -fy*py + -fz*pz)},
        {  0,   0,   0,  1},
    }

    return mat4.mul(mat4.perspective(FOV, aspect, NEAR, FAR), view)
end

------------------------------------------------------------------------------
-- Shader. Vertex pass applies viewProjection; fragment does cheap Lambert
-- against a single directional light plus distance fog.
------------------------------------------------------------------------------
local VERT_SRC = [[
attribute vec3 VertexNormal;
attribute vec3 VertexColor3;

uniform mat4 uViewProj;
uniform vec3 uLight;

varying vec3 vColor;
varying vec3 vWorld;
varying float vShade;

vec4 position(mat4 _t, vec4 vp) {
    vec3 n   = normalize(VertexNormal);
    float l  = max(dot(n, normalize(uLight)), 0.0);
    vShade   = 0.35 + 0.65 * l;
    vColor   = VertexColor3;
    vWorld   = vp.xyz;
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
    vec3  col = vColor * vShade;
    return vec4(mix(col, uSky, fog), 1.0);
}
]]

local VERTEX_FORMAT = {
    {"VertexPosition", "float", 3},
    {"VertexNormal",   "float", 3},
    {"VertexColor3",   "float", 3},
}

------------------------------------------------------------------------------
-- Mesh builders. The quad helper emits two triangles wound so the visible
-- side from +normal direction faces the camera (CCW in screen space) — this
-- lets backface culling stay enabled.
------------------------------------------------------------------------------
local function pushQuad(verts, indices, a, b, c, d, n, col)
    local base = #verts
    verts[#verts+1] = {a[1], a[2], a[3], n[1], n[2], n[3], col[1], col[2], col[3]}
    verts[#verts+1] = {b[1], b[2], b[3], n[1], n[2], n[3], col[1], col[2], col[3]}
    verts[#verts+1] = {c[1], c[2], c[3], n[1], n[2], n[3], col[1], col[2], col[3]}
    verts[#verts+1] = {d[1], d[2], d[3], n[1], n[2], n[3], col[1], col[2], col[3]}
    -- Triangles (A, C, B) and (A, D, C). Pen-and-paper check: for the +Z face
    -- A=(x0,y1,z1), B=(x1,y1,z1), C=(x1,y0,z1), D=(x0,y0,z1), the cross
    -- (C-A) × (B-A) = (0, 0, +h²) — points along +Z, the outward normal.
    indices[#indices+1] = base+1; indices[#indices+1] = base+3; indices[#indices+1] = base+2
    indices[#indices+1] = base+1; indices[#indices+1] = base+4; indices[#indices+1] = base+3
end

local function pushCube(verts, indices, cx, cy, cz, s, col)
    local h = s * 0.5
    local x0, y0, z0 = cx-h, cy-h, cz-h
    local x1, y1, z1 = cx+h, cy+h, cz+h
    pushQuad(verts, indices, {x0,y1,z1},{x1,y1,z1},{x1,y0,z1},{x0,y0,z1}, { 0, 0, 1}, col)  -- +Z
    pushQuad(verts, indices, {x1,y1,z0},{x0,y1,z0},{x0,y0,z0},{x1,y0,z0}, { 0, 0,-1}, col)  -- -Z
    pushQuad(verts, indices, {x0,y1,z0},{x0,y1,z1},{x0,y0,z1},{x0,y0,z0}, {-1, 0, 0}, col)  -- -X
    pushQuad(verts, indices, {x1,y1,z1},{x1,y1,z0},{x1,y0,z0},{x1,y0,z1}, { 1, 0, 0}, col)  -- +X
    pushQuad(verts, indices, {x0,y1,z0},{x1,y1,z0},{x1,y1,z1},{x0,y1,z1}, { 0, 1, 0}, col)  -- +Y
    pushQuad(verts, indices, {x0,y0,z1},{x1,y0,z1},{x1,y0,z0},{x0,y0,z0}, { 0,-1, 0}, col)  -- -Y
end

-- Module-level list of cubes for per-frame AABB collision (XZ rect + top Y).
local cubes = {}

local function buildWorldMesh()
    local verts, indices = {}, {}

    -- Ground quad — CCW from above (+Y normal).
    pushQuad(verts, indices,
        {-GROUND_HALF, 0, -GROUND_HALF},
        { GROUND_HALF, 0, -GROUND_HALF},
        { GROUND_HALF, 0,  GROUND_HALF},
        {-GROUND_HALF, 0,  GROUND_HALF},
        {0, 1, 0},
        {0.22, 0.28, 0.32}
    )

    -- Deterministic pseudo-random scatter (so the layout matches every run
    -- without touching math.randomseed and disturbing the global RNG).
    local seed = 1
    local function rand()
        seed = (seed * 1103515245 + 12345) % 2147483648
        return seed / 2147483648
    end
    for _ = 1, CUBE_COUNT do
        local x = (rand() * 2 - 1) * (GROUND_HALF - 100)
        local z = (rand() * 2 - 1) * (GROUND_HALF - 100)
        local h = CUBE_SIZE * (0.6 + rand() * 1.8)
        local col = {
            0.4 + rand() * 0.55,
            0.4 + rand() * 0.55,
            0.4 + rand() * 0.55,
        }
        pushCube(verts, indices, x, h * 0.5, z, h, col)
        cubes[#cubes+1] = { x = x, z = z, half = h * 0.5, top = h }
    end

    local mesh = love.graphics.newMesh(VERTEX_FORMAT, verts, "triangles", "static")
    mesh:setVertexMap(indices)
    return mesh
end

------------------------------------------------------------------------------
-- Collision. Axis-separated AABB push-out: try X first, then Z. The player
-- is modeled as a cylinder (PLAYER_R radius) and only blocks against cubes
-- whose top edge is above the player's feet — so a short cube the player
-- has jumped above is walked over freely.
------------------------------------------------------------------------------
local function blockedAt(x, z, eyeY)
    local footY = eyeY - EYE_HEIGHT
    for _, c in ipairs(cubes) do
        local dx = x - c.x
        local dz = z - c.z
        local pad = c.half + PLAYER_R
        if dx > -pad and dx < pad and dz > -pad and dz < pad then
            if footY < c.top - 0.01 then return true end
        end
    end
    return false
end

------------------------------------------------------------------------------
-- Love callbacks
------------------------------------------------------------------------------
local world, shader

local SKY = {0.55, 0.70, 0.85}

function love.load()
    love.window.setTitle("Love2D 3D — single-file demo")
    -- Ask Love2D for a 16-bit depth buffer on the main framebuffer so we
    -- can z-test directly against the screen — no intermediate canvas, no
    -- canvas-Y-flip subtleties to worry about.
    love.window.setMode(1280, 720, { resizable = true, vsync = true, depth = 16 })
    love.mouse.setRelativeMode(true)

    shader = love.graphics.newShader(FRAG_SRC, VERT_SRC)
    world  = buildWorldMesh()
end

function love.mousemoved(_, _, dx, dy)
    -- Raw deltas — never multiply by dt; they're already time-independent.
    camera:rotate(dx, dy)
end

function love.keypressed(key)
    if key == "escape" then love.event.quit(0) end
    if key == "space" and camera.onGround then
        camera.velY     = JUMP_SPEED
        camera.onGround = false
    end
end

function love.update(dt)
    local fx, fz = camera:forwardXZ()
    local rx, rz = camera:rightXZ()

    local kb = love.keyboard
    local forward = (kb.isDown("w") and 1 or 0) - (kb.isDown("s") and 1 or 0)
    local side    = (kb.isDown("d") and 1 or 0) - (kb.isDown("a") and 1 or 0)

    local vx = fx * forward + rx * side
    local vz = fz * forward + rz * side
    local len = math.sqrt(vx*vx + vz*vz)
    if len > 0 then vx, vz = vx / len, vz / len end

    -- Axis-separated horizontal move so the player slides along walls.
    local nx = camera.pos[1] + vx * MOVE_SPEED * dt
    local nz = camera.pos[3] + vz * MOVE_SPEED * dt
    if not blockedAt(nx, camera.pos[3], camera.pos[2]) then camera.pos[1] = nx end
    if not blockedAt(camera.pos[1], nz, camera.pos[2]) then camera.pos[3] = nz end

    -- Vertical: simple gravity + ground clamp at eye-height.
    camera.velY  = camera.velY - GRAVITY * dt
    camera.pos[2] = camera.pos[2] + camera.velY * dt
    if camera.pos[2] <= EYE_HEIGHT then
        camera.pos[2]   = EYE_HEIGHT
        camera.velY     = 0
        camera.onGround = true
    end
end

function love.draw()
    local w, h   = love.graphics.getDimensions()
    local aspect = w / h
    local vp     = camera:viewProjection(aspect)

    -- 3D pass — depth test + back-face culling on the main framebuffer.
    love.graphics.clear(SKY[1], SKY[2], SKY[3], 1, true, true)
    love.graphics.setDepthMode("lequal", true)
    love.graphics.setMeshCullMode("back")
    love.graphics.setShader(shader)
    shader:send("uViewProj", "row", vp)
    shader:send("uLight",    {0.4, 1.0, 0.3})
    shader:send("uSky",      SKY)
    shader:send("uFog",      GROUND_HALF * 1.4)
    love.graphics.draw(world)

    -- Reset to 2D for the HUD.
    love.graphics.setShader()
    love.graphics.setDepthMode()
    love.graphics.setMeshCullMode("none")

    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.print(
        ("FPS %d   pos %.0f, %.0f, %.0f"):format(
            love.timer.getFPS(),
            camera.pos[1], camera.pos[2], camera.pos[3]),
        8, 8)
    love.graphics.print("WASD move, mouse look, space jump, esc quit", 8, 24)
end
