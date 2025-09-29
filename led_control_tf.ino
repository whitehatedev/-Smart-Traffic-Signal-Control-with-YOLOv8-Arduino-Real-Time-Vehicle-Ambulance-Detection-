// Define pin numbers for signal lamps
#define NORTH_GREEN 2
#define NORTH_YELLOW 3
#define NORTH_RED 4

#define EAST_GREEN 5
#define EAST_YELLOW 6
#define EAST_RED 7

#define SOUTH_GREEN 8
#define SOUTH_YELLOW 9
#define SOUTH_RED 10

#define WEST_GREEN 11
#define WEST_YELLOW 12
#define WEST_RED 13

// Function to turn off all LEDs
void turnOffAllLEDs() {
    digitalWrite(NORTH_GREEN, LOW);
    digitalWrite(NORTH_YELLOW, LOW);
    digitalWrite(NORTH_RED, LOW);

    digitalWrite(EAST_GREEN, LOW);
    digitalWrite(EAST_YELLOW, LOW);
    digitalWrite(EAST_RED, LOW);

    digitalWrite(SOUTH_GREEN, LOW);
    digitalWrite(SOUTH_YELLOW, LOW);
    digitalWrite(SOUTH_RED, LOW);

    digitalWrite(WEST_GREEN, LOW);
    digitalWrite(WEST_YELLOW, LOW);
    digitalWrite(WEST_RED, LOW);
}

// Function to set a specific direction's signal state
void setSignalState(String direction, String state) {
    turnOffAllLEDs(); // Ensure all LEDs are off before setting a new state

    if (direction == "North") {
        if (state == "Green") digitalWrite(NORTH_GREEN, HIGH);
        else if (state == "Yellow") digitalWrite(NORTH_YELLOW, HIGH);
        else if (state == "Red") digitalWrite(NORTH_RED, HIGH);
    } else if (direction == "East") {
        if (state == "Green") digitalWrite(EAST_GREEN, HIGH);
        else if (state == "Yellow") digitalWrite(EAST_YELLOW, HIGH);
        else if (state == "Red") digitalWrite(EAST_RED, HIGH);
    } else if (direction == "South") {
        if (state == "Green") digitalWrite(SOUTH_GREEN, HIGH);
        else if (state == "Yellow") digitalWrite(SOUTH_YELLOW, HIGH);
        else if (state == "Red") digitalWrite(SOUTH_RED, HIGH);
    } else if (direction == "West") {
        if (state == "Green") digitalWrite(WEST_GREEN, HIGH);
        else if (state == "Yellow") digitalWrite(WEST_YELLOW, HIGH);
        else if (state == "Red") digitalWrite(WEST_RED, HIGH);
    }
}

void setup() {
    // Initialize LED pins as output
    pinMode(NORTH_GREEN, OUTPUT);
    pinMode(NORTH_YELLOW, OUTPUT);
    pinMode(NORTH_RED, OUTPUT);

    pinMode(EAST_GREEN, OUTPUT);
    pinMode(EAST_YELLOW, OUTPUT);
    pinMode(EAST_RED, OUTPUT);

    pinMode(SOUTH_GREEN, OUTPUT);
    pinMode(SOUTH_YELLOW, OUTPUT);
    pinMode(SOUTH_RED, OUTPUT);

    pinMode(WEST_GREEN, OUTPUT);
    pinMode(WEST_YELLOW, OUTPUT);
    pinMode(WEST_RED, OUTPUT);

    // Start serial communication
    Serial.begin(9600);
}

void loop() {
    // Check if data is available on the serial port
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n'); // Read the input string
        input.trim(); // Remove any leading/trailing whitespace

        // Parse the input for direction and state
        int separatorIndex = input.indexOf(',');
        if (separatorIndex != -1) {
            String direction = input.substring(0, separatorIndex);
            String state = input.substring(separatorIndex + 1);
            
            setSignalState(direction, state);
        }
    }
}
