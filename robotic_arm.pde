//-------------------GLOBAL DEFINITIONS AND SETUP-------------------//
// One window with three panels: the main 3D view, a view from above (to judge
// depth), and the camera preview streamed by hand_detector.py.

import processing.net.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import javax.imageio.ImageIO;

PShape base, shoulder, upArm, loArm, end, table1, cube;

// Views: each is rendered off-screen, then placed in the window
PGraphics mainView, topView;
float mainRotX = 0, mainRotY = 0;     // front view, like the original sketch
float topRotX = -0.75, topRotY = 0.55; // elevated view from the side
int dragging = -1;                    // which view the mouse is rotating: 0 main, 1 top
final int M = 8;                      // gap between panels, in pixels
int[] mainRect, topRect, camRect;     // x, y, w, h of each panel

// Arm: link lengths and joint angles (3 joints: base yaw, shoulder, elbow)
final float F = 50;
final float T = 70;
float alpha, beta, gamma;

// Target from the hand tracker, in arm coordinates. With the front view:
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

// Network: control lines on serverPort, camera frames on serverPort + 1
Client client, video;
String serverIP = "127.0.0.1";
int serverPort = 65432;
int lastConnectAttempt = 0, lastVideoAttempt = 0;
final int RECONNECT_MS = 2000;
byte[] videoBuffer = new byte[0];
PImage cameraImage;
int lastFrameMillis = -100000;
String toast = "";
int toastUntil = 0;

void settings() {
  // Fill most of the screen; the window can be resized or maximized
  size(max(960, min(1600, displayWidth - 80)), max(600, min(900, displayHeight - 140)), P3D);
  smooth(4);
}

void setup() {
  surface.setResizable(true);
  surface.setTitle("Visuocontrol");
  surface.setLocation(max(0, (displayWidth - width) / 2), max(0, (displayHeight - height) / 2 - 30));
  textFont(createFont("SansSerif", 14));

  initializeShapes();
  connect();
  connectVideo();
  lastMillis = millis();
}

//-------------------MAIN LIFECYCLE FUNCTIONS-------------------//

void draw() {
  int now = millis();
  float dt = min((now - lastMillis) / 1000.0, 0.05); // cap the step if a frame stalls
  lastMillis = now;

  handleClientData();
  handleVideo();
  IK();
  updateCube(dt);

  layoutPanels();
  mainView = renderView(mainView, mainRect, mainRotX, mainRotY, 1.2);
  topView = renderView(topView, topRect, topRotX, topRotY, 1.0);

  background(18);
  hint(DISABLE_DEPTH_TEST); // everything below is flat 2D on top of the views
  image(mainView, mainRect[0], mainRect[1]);
  image(topView, topRect[0], topRect[1]);
  drawCameraPanel();
  panelLabel(mainRect, "Front view");
  panelLabel(topRect, "View from above: depth");
  panelLabel(camRect, "Camera");
  drawStatusBar();
  hint(ENABLE_DEPTH_TEST);
}

void mousePressed() {
  dragging = inside(mainRect, mouseX, mouseY) ? 0 : inside(topRect, mouseX, mouseY) ? 1 : -1;
}

void mouseReleased() {
  dragging = -1;
}

void mouseDragged() {
  float dx = (mouseX - pmouseX) * 0.01, dy = (mouseY - pmouseY) * 0.01;
  if (dragging == 0) { mainRotY -= dx; mainRotX -= dy; }
  if (dragging == 1) { topRotY -= dx; topRotX -= dy; }
}

void keyPressed() {
  if (key == 'r' || key == 'R') {
    resetCube();
    showToast("Cube reset");
  }
  if (key == 'c' || key == 'C') {
    if (client != null && client.active()) {
      client.write("recenter\n"); // the tracker re-centers depth at the hand's current distance
      showToast("Depth re-centered at your hand's current distance");
    } else {
      showToast("Hand tracker not connected");
    }
  }
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

//-------------------LAYOUT AND 2D PANELS-------------------//

void layoutPanels() {
  int colW = constrain(int(width * 0.32), 300, 560);
  int colH = (height - 3 * M) / 2;
  mainRect = new int[] {M, M, width - colW - 3 * M, height - 2 * M};
  topRect = new int[] {width - colW - M, M, colW, colH};
  camRect = new int[] {width - colW - M, 2 * M + colH, colW, height - 3 * M - colH};
}

boolean inside(int[] r, int x, int y) {
  return x >= r[0] && x < r[0] + r[2] && y >= r[1] && y < r[1] + r[3];
}

// Draw the scene into an off-screen buffer the size of the panel
PGraphics renderView(PGraphics pg, int[] r, float rx, float ry, float zoom) {
  if (pg == null || pg.width != r[2] || pg.height != r[3]) {
    pg = createGraphics(max(1, r[2]), max(1, r[3]), P3D);
  }
  pg.beginDraw();
  drawWorld(pg, rx, ry, zoom);
  pg.endDraw();
  return pg;
}

void drawCameraPanel() {
  int[] r = camRect;
  noStroke();
  fill(26);
  rect(r[0], r[1], r[2], r[3]);
  boolean fresh = cameraImage != null && millis() - lastFrameMillis < 2000;
  if (fresh) {
    // Fit the image inside the panel, keeping its aspect ratio
    float s = min(r[2] / (float) cameraImage.width, r[3] / (float) cameraImage.height);
    float w = cameraImage.width * s, h = cameraImage.height * s;
    image(cameraImage, r[0] + (r[2] - w) / 2, r[1] + (r[3] - h) / 2, w, h);
  } else {
    fill(150);
    textAlign(CENTER, CENTER);
    boolean tracker = client != null && client.active();
    text(tracker ? "Waiting for the camera..." : "Hand tracker not running.\nStart it with Launch.bat",
         r[0] + r[2] / 2, r[1] + r[3] / 2);
    textAlign(LEFT, BASELINE);
  }
}

void panelLabel(int[] r, String label) {
  noStroke();
  fill(0, 150);
  rect(r[0], r[1], textWidth(label) + 16, 24);
  fill(230);
  text(label, r[0] + 8, r[1] + 17);
}

void drawStatusBar() {
  int[] r = mainRect;
  int h = 34, y = r[1] + r[3] - h;
  noStroke();
  fill(0, 170);
  rect(r[0], y, r[2], h);

  boolean tracker = client != null && client.active();
  String state;
  if (!tracker) {
    fill(255, 110, 90);
    state = "Hand tracker not connected";
  } else if (holding) {
    fill(110, 230, 120);
    state = "Holding the cube";
  } else if (grabbed) {
    fill(255, 190, 80);
    state = "Pinching (not at the cube)";
  } else {
    fill(220);
    state = "Pinch at the cube to grab it";
  }
  text(state, r[0] + 12, y + 22);

  fill(170);
  String help = "c: re-center depth    r: reset cube    drag: rotate a view";
  text(help, r[0] + r[2] - textWidth(help) - 12, y + 22);

  if (millis() < toastUntil) {
    float w = textWidth(toast) + 24;
    fill(0, 200);
    rect(r[0] + (r[2] - w) / 2, r[1] + 14, w, 30, 6);
    fill(255);
    textAlign(CENTER, CENTER);
    text(toast, r[0] + r[2] / 2, r[1] + 29);
    textAlign(LEFT, BASELINE);
  }
}

void showToast(String message) {
  toast = message;
  toastUntil = millis() + 1800;
}

//-------------------3D RENDERING FUNCTIONS-------------------//
// Both views draw the same scene through these functions; p is the view's buffer.

void drawWorld(PGraphics p, float rx, float ry, float zoom) {
  p.background(32);
  p.lights();
  p.directionalLight(51, 102, 126, -1, 0, 0);
  p.noStroke();

  p.translate(p.width / 2, p.height * 0.55);
  p.rotateX(rx);
  p.rotateY(-ry);
  p.scale(-4 * zoom * min(p.width, p.height) / 800.0); // the original 800x800 window used scale(-4)

  drawFloor(p);
  drawShadows(p);
  drawCube(p);
  drawTables(p);
  drawRoboticArm(p);
}

void drawFloor(PGraphics p) {
  p.pushMatrix();
  p.translate(0, FLOOR_Y - 0.5, 0);
  p.fill(78);
  p.box(260, 1, 160);
  p.popMatrix();
}

// Dark squares straight below the cube and the arm tip, on whatever surface is
// underneath. They make it much easier to judge depth with one hand.
void drawShadows(PGraphics p) {
  p.fill(20);
  float cubeGround = supportHeight(posX, posZ);
  if (posY > cubeGround + 0.5) {
    flatSquare(p, posX, cubeGround, posZ, CUBE_HALF * 2);
  }
  p.fill(200, 60, 200);
  flatSquare(p, tipX, supportHeight(tipX, tipZ), tipZ, 4);
}

void flatSquare(PGraphics p, float x, float y, float z, float size) {
  p.pushMatrix();
  p.translate(x, y + 0.15, z);
  p.box(size, 0.1, size);
  p.popMatrix();
}

void drawTables(PGraphics p) {
  p.pushMatrix();
  p.scale(0.25);
  p.shape(table1, -250, -200);
  p.popMatrix();

  p.pushMatrix();
  p.scale(0.25);
  p.shape(table1, 250, -200);
  p.popMatrix();
}

void drawCube(PGraphics p) {
  p.pushMatrix();
  p.translate(posX, posY, posZ);
  p.scale(0.75);
  p.shape(cube);
  p.popMatrix();
}

void drawRoboticArm(PGraphics p) {
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

void connectVideo() {
  video = new Client(this, serverIP, serverPort + 1);
  videoBuffer = new byte[0];
  lastVideoAttempt = millis();
}

// Camera frames arrive as a 4-byte big-endian length followed by a JPEG.
// Collect bytes, cut out whole frames, and show only the newest one.
void handleVideo() {
  if (video == null || !video.active()) {
    if (millis() - lastVideoAttempt > RECONNECT_MS) connectVideo();
    return;
  }
  if (video.available() > 0) {
    videoBuffer = concat(videoBuffer, video.readBytes());
  }
  byte[] latest = null;
  int off = 0;
  while (videoBuffer.length - off >= 4) {
    int n = ((videoBuffer[off] & 0xff) << 24) | ((videoBuffer[off + 1] & 0xff) << 16)
          | ((videoBuffer[off + 2] & 0xff) << 8) | (videoBuffer[off + 3] & 0xff);
    if (n <= 0 || n > 5000000) { // out of sync: start over with the next data
      videoBuffer = new byte[0];
      return;
    }
    if (videoBuffer.length - off - 4 < n) break; // frame not complete yet
    latest = subset(videoBuffer, off + 4, n);
    off += 4 + n;
  }
  if (off > 0) videoBuffer = subset(videoBuffer, off);
  if (latest != null) decodeFrame(latest);
}

void decodeFrame(byte[] jpg) {
  try {
    BufferedImage img = ImageIO.read(new ByteArrayInputStream(jpg));
    if (img == null) return;
    int w = img.getWidth(), h = img.getHeight();
    if (cameraImage == null || cameraImage.width != w || cameraImage.height != h) {
      cameraImage = createImage(w, h, RGB);
    }
    cameraImage.loadPixels();
    img.getRGB(0, 0, w, h, cameraImage.pixels, 0, w);
    cameraImage.updatePixels();
    lastFrameMillis = millis();
  }
  catch (IOException e) {
    // a damaged frame: skip it
  }
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
