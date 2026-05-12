-- Tiny 3D demo in Love2D — one file, no deps.
-- A flat ground plane with colored cubes you can walk around.
--
-- The whole thing is: a hand-rolled row-major mat4, a yaw/pitch camera,
-- and one shader. Love2D's 2D API renders 3D fine once you attach a depth
-- buffer to a canvas and ship a viewProjection matrix to a custom shader.
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
local EYE_HEIGHT  = 64       -- camera Y when grounded
local GRAVITY     = 1400     -- world units / s² pulling player down
local JUMP_SPEED  = 480      -- initial upward velocity on space
local PLAYER_R    = 18       -- cylinder radius for cube collision

local GROUND_HALF = 1200
local CUBE_SIZE   = 40
local CUBE_COUNT  = 64

------------------------------------------------------------------------------
-- mat4 — row-major. Send to shaders with :send(name, "row", m).
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
-- Camera. Yaw around Y, pitch around X (clamped to dodge gimbal flip).
------------------------------------------------------------------------------
local camera = {
    pos    = {0, EYE_HEIGHT, 400},
    yaw    = 0,
    pitch  = 0,
    velY   = 0,        -- vertical velocity (for jump/gravity)
    onGround = true,
}

function camera:rotate(dx, dy)
    self.yaw   = self.yaw   + dx * MOUSE_SENS
    self.pitch = self.pitch - dy * MOUSE_SENS
    local lim  = math.pi * 0.5 - 0.01
    if self.pitch >  lim then self.pitch =  lim end
    if self.pitch < -lim then self.pitch = -lim end
end

-- Right-handed, +X right, +Y up, -Z forward at yaw=0.
-- Mouse right (dx > 0) → yaw increases → forward turns toward +X.
function camera:forwardXZ()
    return math.sin(self.yaw), -math.cos(self.yaw)
end

function camera:rightXZ()
    return math.cos(self.yaw), math.sin(self.yaw)
end

function camera:viewProjection(aspect)
    local p  = mat4.perspective(FOV, aspect, NEAR, FAR)
    local ry = mat4.rotationY(self.yaw)
    local rx = mat4.rotationX(-self.pitch)
    local t  = mat4.translation(-self.pos[1], -self.pos[2], -self.pos[3])
    return mat4.mul(p, mat4.mul(rx, mat4.mul(ry, t)))
end

------------------------------------------------------------------------------
-- Shader. Vertex pass applies viewProjection; fragment does Lambert + fog.
-- Normals are baked into the mesh per-vertex (flat: all 4 verts of a face
-- share the face normal) so the shader stays trivial.
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

-- Vertex layout: position (3) + normal (3) + color (3). Floats.
local VERTEX_FORMAT = {
    {"VertexPosition", "float", 3},
    {"VertexNormal",   "float", 3},
    {"VertexColor3",   "float", 3},
}

------------------------------------------------------------------------------
-- Mesh builders
------------------------------------------------------------------------------

-- Append two CCW triangles for a quad given 4 corners + face normal + color.
local function pushQuad(verts, indices, a, b, c, d, n, col)
    local base = #verts
    verts[#verts+1] = {a[1], a[2], a[3], n[1], n[2], n[3], col[1], col[2], col[3]}
    verts[#verts+1] = {b[1], b[2], b[3], n[1], n[2], n[3], col[1], col[2], col[3]}
    verts[#verts+1] = {c[1], c[2], c[3], n[1], n[2], n[3], col[1], col[2], col[3]}
    verts[#verts+1] = {d[1], d[2], d[3], n[1], n[2], n[3], col[1], col[2], col[3]}
    -- CCW seen from the +normal side.
    indices[#indices+1] = base+1; indices[#indices+1] = base+2; indices[#indices+1] = base+3
    indices[#indices+1] = base+1; indices[#indices+1] = base+3; indices[#indices+1] = base+4
end

local function pushCube(verts, indices, cx, cy, cz, s, col)
    local h = s * 0.5
    local x0,y0,z0 = cx-h, cy-h, cz-h
    local x1,y1,z1 = cx+h, cy+h, cz+h
    -- 6 faces. Quad corner order chosen so CCW points along the named normal.
    pushQuad(verts, indices, {x0,y1,z1},{x1,y1,z1},{x1,y0,z1},{x0,y0,z1}, { 0, 0, 1}, col)  -- front  (+Z)
    pushQuad(verts, indices, {x1,y1,z0},{x0,y1,z0},{x0,y0,z0},{x1,y0,z0}, { 0, 0,-1}, col)  -- back   (-Z)
    pushQuad(verts, indices, {x0,y1,z0},{x0,y1,z1},{x0,y0,z1},{x0,y0,z0}, {-1, 0, 0}, col)  -- left   (-X)
    pushQuad(verts, indices, {x1,y1,z1},{x1,y1,z0},{x1,y0,z0},{x1,y0,z1}, { 1, 0, 0}, col)  -- right  (+X)
    pushQuad(verts, indices, {x0,y1,z0},{x1,y1,z0},{x1,y1,z1},{x0,y1,z1}, { 0, 1, 0}, col)  -- top    (+Y)
    pushQuad(verts, indices, {x0,y0,z1},{x1,y0,z1},{x1,y0,z0},{x0,y0,z0}, { 0,-1, 0}, col)  -- bottom (-Y)
end

-- Module-level list of cubes used by both the mesh builder and the
-- per-frame collision check. Each entry is {x, z, halfSize, top}.
local cubes = {}

local function buildWorldMesh()
    local verts, indices = {}, {}

    -- Ground quad. CCW seen from above (+Y).
    pushQuad(verts, indices,
        {-GROUND_HALF, 0, -GROUND_HALF},
        { GROUND_HALF, 0, -GROUND_HALF},
        { GROUND_HALF, 0,  GROUND_HALF},
        {-GROUND_HALF, 0,  GROUND_HALF},
        {0, 1, 0},
        {0.22, 0.28, 0.32}
    )

    -- Deterministic pseudo-random scatter so the layout is stable across
    -- runs (no math.randomseed touching the global RNG state).
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
        cubes[#cubes + 1] = { x = x, z = z, half = h * 0.5, top = h }
    end

    local mesh = love.graphics.newMesh(VERTEX_FORMAT, verts, "triangles", "static")
    mesh:setVertexMap(indices)
    return mesh
end

-- Axis-separated AABB push-out: try the X move alone, then the Z move alone,
-- so the player slides along walls instead of getting stuck on a corner.
local function blockedAt(x, z, eyeY)
    local footY = eyeY - EYE_HEIGHT
    for _, c in ipairs(cubes) do
        local dx = x - c.x
        local dz = z - c.z
        local pad = c.half + PLAYER_R
        if dx > -pad and dx < pad and dz > -pad and dz < pad then
            -- XZ overlaps. Only block if the body actually straddles the cube
            -- vertically — eye below cube top AND foot below cube top means
            -- we'd intersect the box. If feet are above the cube's top, the
            -- player is on/over it and can pass freely.
            if footY < c.top - 0.01 then return true end
        end
    end
    return false
end

------------------------------------------------------------------------------
-- Love callbacks
------------------------------------------------------------------------------
local world, shader, canvas

local SKY = {0.55, 0.70, 0.85}

local function newCanvas(w, h)
    local c = love.graphics.newCanvas(w, h, { format = "rgba8" })
    c:setFilter("linear", "linear")
    return c
end

function love.load()
    love.window.setTitle("Love2D 3D — single-file demo")
    love.window.setMode(1280, 720, { resizable = true, vsync = true })
    love.mouse.setRelativeMode(true)

    shader = love.graphics.newShader(FRAG_SRC, VERT_SRC)
    world  = buildWorldMesh()
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
    if key == "space" and camera.onGround then
        camera.velY    = JUMP_SPEED
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

    -- Horizontal move, axis-separated for wall sliding.
    local dx = vx * MOVE_SPEED * dt
    local dz = vz * MOVE_SPEED * dt
    local nx = camera.pos[1] + dx
    local nz = camera.pos[3] + dz
    if not blockedAt(nx, camera.pos[3], camera.pos[2]) then camera.pos[1] = nx end
    if not blockedAt(camera.pos[1], nz, camera.pos[2]) then camera.pos[3] = nz end

    -- Gravity / jump arc. Ground at Y = EYE_HEIGHT (camera's eye sits at
    -- that height when feet touch the plane).
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

    -- 3D pass: depth-attached canvas, depth test on, back-face culling on.
    love.graphics.setCanvas({ canvas, depth = true })
    love.graphics.clear(SKY[1], SKY[2], SKY[3], 1, true, true)
    love.graphics.setDepthMode("lequal", true)
    love.graphics.setMeshCullMode("back")
    love.graphics.setShader(shader)
    shader:send("uViewProj", "row", vp)
    shader:send("uLight",    {0.4, 1.0, 0.3})
    shader:send("uSky",      SKY)
    shader:send("uFog",      GROUND_HALF * 1.4)
    love.graphics.draw(world)

    -- Back to 2D for the HUD.
    love.graphics.setShader()
    love.graphics.setDepthMode()
    love.graphics.setMeshCullMode("none")
    love.graphics.setCanvas()

    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.draw(canvas, 0, 0)

    love.graphics.print(
        ("FPS %d   pos %.0f, %.0f, %.0f"):format(
            love.timer.getFPS(),
            camera.pos[1], camera.pos[2], camera.pos[3]),
        8, 8)
    love.graphics.print("WASD move, mouse look, space jump, esc quit", 8, 24)
end
