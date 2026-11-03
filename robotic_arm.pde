//-------------------GLOBAL DEFINITIONS AND SETUP-------------------//

import processing.net.*;

PShape base, shoulder, upArm, loArm, end, table1, cube;
float rotX, rotY;
float posX = -60, posY = -37, posZ = 0; // Initial position of the cube
float alpha, beta, gamma;
float F = 50;
float T = 70;
Client client;
String serverIP = "127.0.0.1";
int serverPort = 65432;
int lastConnectAttempt = 0;
final int RECONNECT_MS = 2000;
float coordX, coordY, coordZ;
String mano_cerrada = "False";
boolean grabbed = false;

void setup() {
  size(800, 800, OPENGL);

  surface.setResizable(true); // Make the window resizable

  initializeShapes();

  connect();

  // Open a duplicate window
  DuplicateWindow duplicate = new DuplicateWindow();
  PApplet.runSketch(new String[]{"DuplicateWindow"}, duplicate);
}

//-------------------MAIN LIFECYCLE FUNCTIONS-------------------//

void draw() {
  background(32);
  handleClientData();

  if (grabbed && isArmTouchingCube()) {
    posX = -coordY;
    posY = -coordZ;
    posZ = -coordX;
  }

  IK();
  renderScene();
}

void mouseDragged() {
  rotY -= (mouseX - pmouseX) * 0.01;
  rotX -= (mouseY - pmouseY) * 0.01;
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

//-------------------RENDERING FUNCTIONS-------------------//

void renderScene() {
  smooth();
  lights();
  directionalLight(51, 102, 126, -1, 0, 0);
  noStroke();

  translate(width / 2, height / 2);
  rotateX(rotX);
  rotateY(-rotY);
  scale(-4); // Added scaling here for consistency

  renderCube();
  renderTables();
  renderRoboticArm();
}

void renderTables() {
  pushMatrix();
  scale(0.25);
  shape(table1, -250, -200);
  popMatrix();

  pushMatrix();
  scale(0.25);
  shape(table1, 250, -200);
  popMatrix();
}

void renderCube() {
  pushMatrix();
  translate(posX, posY, posZ);
  scale(0.75);
  shape(cube);
  popMatrix();
}

void renderRoboticArm() {
  fill(#FFE308);
  translate(0, -40, 0);
  if (base != null) shape(base);

  translate(0, 4, 0);
  rotateY(gamma);
  if (shoulder != null) shape(shoulder);

  translate(0, 25, 0);
  rotateY(PI);
  rotateX(alpha);
  if (upArm != null) shape(upArm);

  translate(0, 0, 50);
  rotateY(PI);
  rotateX(beta);
  if (loArm != null) shape(loArm);

  translate(0, 0, -50);
  rotateY(PI);
  if (end != null) shape(end);
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
  // Parse coordinates received from the server
  if (data != null && data.length() > 0) {
    String[] parts = split(data, ',');
    if (parts.length == 4) {
      coordX = float(trim(parts[0]));
      coordY = float(trim(parts[1]));
      coordZ = float(trim(parts[2]));
      mano_cerrada = trim(parts[3]);
      grabbed = mano_cerrada.equals("True");
    }
  }
}

//-------------------KINEMATICS AND LOGIC FUNCTIONS-------------------//

void IK() {
  float X = coordX;
  float Y = coordY;
  float Z = coordZ;

  float L = sqrt(Y * Y + X * X);
  float dia = sqrt(Z * Z + L * L);

  // The arm can only reach distances between |T - F| and T + F. Outside that
  // range acos() returns NaN and the arm disappears, so clamp the distance:
  // an unreachable target leaves the arm fully stretched toward it.
  dia = constrain(dia, abs(T - F) + 0.01, T + F - 0.01);

  alpha = PI / 2 - (atan2(L, Z) + acos(constrain((T * T - F * F - dia * dia) / (-2 * F * dia), -1, 1)));
  beta = -PI + acos(constrain((dia * dia - T * T - F * F) / (-2 * F * T), -1, 1));
  gamma = atan2(Y, X);
}

boolean isArmTouchingCube() {
  float tolerance = 30.0; // Define the tolerance value

  // Calculate the difference between the coordinates of the arm and the cube
  float diffX = abs(coordX + posZ);
  float diffY = abs(coordY + posX);
  float diffZ = abs(coordZ + posY);

  // Check if all differences are within the tolerance
  boolean isTouching = (diffX < tolerance) && (diffY < tolerance) && (diffZ < tolerance);

  return isTouching;
}

//-------------------DUPLICATE WINDOW CLASS-------------------//

public class DuplicateWindow extends PApplet {
  float rotX, rotY; // Mismo estado de la cámara que en la ventana principal

  public void settings() {
    size(800, 800, OPENGL);
  }

  public void setup() {
    surface.setResizable(true); // Make the window resizable
    initializeShapes();
  }

  public void draw() {
    background(32);
    renderScene(rotX, rotY);
  }

  public void mouseDragged() {
    rotY -= (mouseX - pmouseX) * 0.01;
    rotX -= (mouseY - pmouseY) * 0.01;
  }

  void renderScene(float rotX, float rotY) {
    smooth();
    lights();
    directionalLight(51, 102, 126, -1, 0, 0);

    noStroke();

    translate(width / 2, height / 2);
    rotateX(rotX);
    rotateY(-rotY);
    scale(-4);

    pushMatrix();
    scale(0.25);
    shape(table1, -250, -200);
    popMatrix();
    pushMatrix();
    scale(0.25);
    shape(table1, 250, -200);
    popMatrix();

    pushMatrix();
    translate(posX, posY, posZ);
    scale(0.75);
    shape(cube);
    popMatrix();

    fill(#FFE308);
    translate(0, -40, 0);
    if (base != null) shape(base);

    translate(0, 4, 0);
    rotateY(gamma);
    if (shoulder != null) shape(shoulder);

    translate(0, 25, 0);
    rotateY(PI);
    rotateX(alpha);
    if (upArm != null) shape(upArm);

    translate(0, 0, 50);
    rotateY(PI);
    rotateX(beta);
    if (loArm != null) shape(loArm);

    translate(0, 0, -50);
    rotateY(PI);
    if (end != null) shape(end);
  }
}
