//-------------------GLOBAL DEFINITIONS AND SETUP-------------------//

import processing.net.*;

PShape base, shoulder, upArm, loArm, end, table1, cube;
float rotX, rotY; // camera rotation of the main window (mouse drag)

// Arm: link lengths and joint angles (3 joints: base yaw, shoulder, elbow)
final float F = 50;
final float T = 70;
float alpha, beta, gamma;

// Target from the hand tracker, in arm coordinates. With the default view:
// coordX points toward the viewer, coordY to the right, coordZ down.
float coordX, coordY, coordZ;
boolean grabbed = false;    // the hand is pinching
boolean wasGrabbed = false;
float tipX, tipY, tipZ;     // where the arm tip actually is, in scene coordinates

// Scene geometry in scene coordinates (+Y is up), measured from table.obj and cube.obj
final float TABLE_TOP = -36.9;
final float FLOOR_Y = -62.5;
final float TABLE_X_MIN = 45, TABLE_X_MAX = 80, TABLE_Z_HALF = 27.5; // one table on each side
final float CUBE_HEIGHT = 13.8, CUBE_HALF = 6.85;

// Cube state: position of the centre of its bottom face
final float CUBE_START_X = -60, CUBE_START_Y = -37, CUBE_START_Z = 0;
float posX = CUBE_START_X, posY = CUBE_START_Y, posZ = CUBE_START_Z;
float velY = 0;
boolean holding = false;
float holdDX, holdDY, holdDZ; // cube offset from the tip while it is held

final float GRAB_RADIUS = 14; // tip must be this close to the cube's centre to pick it up
final float GRAVITY = 1500;   // scene units per second squared
final float BOUNCE = 0.25;    // fraction of speed kept when the cube lands
int lastMillis;

// Network
Client client;
String serverIP = "127.0.0.1";
int serverPort = 65432;
int lastConnectAttempt = 0;
final int RECONNECT_MS = 2000;

void setup() {
  size(800, 800, P3D);
  surface.setResizable(true);
  surface.setTitle("robotic_arm (r: reset cube)");

  initializeShapes();
  connect();
  lastMillis = millis();

  // Second window with its own camera angle
  DuplicateWindow duplicate = new DuplicateWindow();
  PApplet.runSketch(new String[]{"DuplicateWindow"}, duplicate);
}

//-------------------MAIN LIFECYCLE FUNCTIONS-------------------//

void draw() {
  int now = millis();
  float dt = min((now - lastMillis) / 1000.0, 0.05); // cap the step if a frame stalls
  lastMillis = now;

  handleClientData();
  IK();
  updateCube(dt);
  drawWorld(this, rotX, rotY);
}

void mouseDragged() {
  rotY -= (mouseX - pmouseX) * 0.01;
  rotX -= (mouseY - pmouseY) * 0.01;
}

void keyPressed() {
  if (key == 'r' || key == 'R') resetCube();
}

//-------------------INITIALIZATION FUNCTIONS-------------------//

void initializeShapes() {
  base = loadShape("r5.obj");
  shoulder = loadShape("r1.obj");
  upArm = loadShape("r2.obj");
  loArm = loadShape("r3.obj");
  end = loadShape("r4.obj");
  table1 = loadShape("table.obj");
  cube = loadShape("cube.obj");

  if (shoulder != null) shoulder.disableStyle();
  if (upArm != null) upArm.disableStyle();
  if (loArm != null) loArm.disableStyle();
}

void resetCube() {
  posX = CUBE_START_X;
  posY = CUBE_START_Y;
  posZ = CUBE_START_Z;
  velY = 0;
  holding = false;
}

//-------------------RENDERING FUNCTIONS-------------------//
// Both windows draw the same scene through these functions; p is the window.

void drawWorld(PApplet p, float rx, float ry) {
  p.background(32);
  p.lights();
  p.directionalLight(51, 102, 126, -1, 0, 0);
  p.noStroke();

  p.translate(p.width / 2, p.height / 2);
  p.rotateX(rx);
  p.rotateY(-ry);
  p.scale(-4);

  drawFloor(p);
  drawShadows(p);
  drawCube(p);
  drawTables(p);
  drawRoboticArm(p);
}

void drawFloor(PApplet p) {
  p.pushMatrix();
  p.translate(0, FLOOR_Y - 0.5, 0);
  p.fill(48);
  p.box(260, 1, 160);
  p.popMatrix();
}

// Dark squares straight below the cube and the arm tip, on whatever surface is
// underneath. They make it much easier to judge depth with one hand.
void drawShadows(PApplet p) {
  p.fill(15, 15, 15);
  float cubeGround = supportHeight(posX, posZ);
  if (posY > cubeGround + 0.5) {
    flatSquare(p, posX, cubeGround, posZ, CUBE_HALF * 2);
  }
  p.fill(120, 40, 120);
  flatSquare(p, tipX, supportHeight(tipX, tipZ), tipZ, 3);
}

void flatSquare(PApplet p, float x, float y, float z, float size) {
  p.pushMatrix();
  p.translate(x, y + 0.15, z);
  p.box(size, 0.1, size);
  p.popMatrix();
}

void drawTables(PApplet p) {
  p.pushMatrix();
  p.scale(0.25);
  p.shape(table1, -250, -200);
  p.popMatrix();

  p.pushMatrix();
  p.scale(0.25);
  p.shape(table1, 250, -200);
  p.popMatrix();
}

void drawCube(PApplet p) {
  p.pushMatrix();
  p.translate(posX, posY, posZ);
  p.scale(0.75);
  p.shape(cube);
  p.popMatrix();
}

void drawRoboticArm(PApplet p) {
  p.fill(#FFE308);
  p.translate(0, -40, 0);
  if (base != null) p.shape(base);

  p.translate(0, 4, 0);
  p.rotateY(gamma);
  if (shoulder != null) p.shape(shoulder);

  p.translate(0, 25, 0);
  p.rotateY(PI);
  p.rotateX(alpha);
  if (upArm != null) p.shape(upArm);

  p.translate(0, 0, 50);
  p.rotateY(PI);
  p.rotateX(beta);
  if (loArm != null) p.shape(loArm);

  p.translate(0, 0, -50);
  p.rotateY(PI);
  if (end != null) p.shape(end);
}

//-------------------CLIENT COMMUNICATION FUNCTIONS-------------------//

void connect() {
  client = new Client(this, serverIP, serverPort);
  lastConnectAttempt = millis();
}

void handleClientData() {
  // Retry every couple of seconds, so the sketch and the hand tracker
  // can be started in any order and either one can be restarted.
  if (client == null || !client.active()) {
    if (millis() - lastConnectAttempt > RECONNECT_MS) connect();
    return;
  }
  // The tracker sends one "x,y,z,grab" line per frame. TCP can split or merge
  // those lines, so read whole lines only and keep the newest one.
  String latest = null;
  while (client.available() > 0) {
    String line = client.readStringUntil('\n');
    if (line == null) break; // partial line: wait for the rest
    latest = line;
  }
  if (latest != null) {
    parseCoordinates(latest);
  }
}

void parseCoordinates(String data) {
  String[] parts = split(trim(data), ',');
  if (parts.length == 4) {
    coordX = float(parts[0]);
    coordY = float(parts[1]);
    coordZ = float(parts[2]);
    grabbed = parts[3].equals("True");
  }
}

//-------------------KINEMATICS AND LOGIC FUNCTIONS-------------------//

void IK() {
  float X = coordX;
  float Y = coordY;
  float Z = coordZ;

  float L = sqrt(Y * Y + X * X);
  float target = sqrt(Z * Z + L * L);

  // The arm can only reach distances between |T - F| and T + F. Outside that
  // range acos() returns NaN and the arm disappears, so clamp the distance:
  // an unreachable target leaves the arm fully stretched toward it.
  float dia = constrain(target, abs(T - F) + 0.01, T + F - 0.01);

  alpha = PI / 2 - (atan2(L, Z) + acos(constrain((T * T - F * F - dia * dia) / (-2 * F * dia), -1, 1)));
  beta = -PI + acos(constrain((dia * dia - T * T - F * F) / (-2 * F * T), -1, 1));
  gamma = atan2(Y, X);

  // Arm coordinates map to scene coordinates as (-Y, -Z, -X). When the target
  // is out of reach, the tip stops short along the same direction.
  float k = target > 0 ? dia / target : 0;
  tipX = -Y * k;
  tipY = -Z * k;
  tipZ = -X * k;
}

// Height the cube rests on at (x, z): a table top or the floor
float supportHeight(float x, float z) {
  boolean overTable = abs(z) <= TABLE_Z_HALF && abs(x) >= TABLE_X_MIN && abs(x) <= TABLE_X_MAX;
  return overTable ? TABLE_TOP : FLOOR_Y;
}

boolean tipNearCube() {
  return dist(tipX, tipY, tipZ, posX, posY + CUBE_HEIGHT / 2, posZ) < GRAB_RADIUS;
}

void updateCube(float dt) {
  // Picking up needs the pinch to start while the tip is at the cube, like a
  // real gripper: pinching first and then touching the cube does nothing.
  boolean pinchStarted = grabbed && !wasGrabbed;
  wasGrabbed = grabbed;

  if (!holding && pinchStarted && tipNearCube()) {
    holding = true;
    holdDX = posX - tipX;
    holdDY = posY - tipY;
    holdDZ = posZ - tipZ;
    velY = 0;
  } else if (holding && !grabbed) {
    holding = false; // let go: gravity takes over below
    velY = 0;
  }

  float ground = supportHeight(posX, posZ);

  if (holding) {
    posX = tipX + holdDX;
    posY = tipY + holdDY;
    posZ = tipZ + holdDZ;
    posY = max(posY, ground); // the cube can't be pushed through a table or the floor
    return;
  }

  if (posY > ground || velY != 0) {
    velY -= GRAVITY * dt;
    posY += velY * dt;
    if (posY <= ground) {
      posY = ground;
      velY = -velY * BOUNCE;
      if (abs(velY) < 40) velY = 0; // settle instead of bouncing forever
    }
  }
}

//-------------------SECOND WINDOW-------------------//

public class DuplicateWindow extends PApplet {
  float rotX = -0.5, rotY = 0.6; // its own camera, starting from another angle

  public void settings() {
    size(800, 800, P3D);
  }

  public void setup() {
    surface.setResizable(true);
    surface.setTitle("robotic_arm (second view)");
    initializeShapes();
  }

  public void draw() {
    drawWorld(this, rotX, rotY);
  }

  public void mouseDragged() {
    rotY -= (mouseX - pmouseX) * 0.01;
    rotX -= (mouseY - pmouseY) * 0.01;
  }
}
