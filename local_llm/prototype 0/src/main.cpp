/*
 * ESP32 BME280 BLE Humidity Monitor
 * Connects to iPhone LightBlue app
 * Sends humidity readings on demand or at regular intervals
 */

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ===== CONFIGURATION =====
const unsigned long UPDATE_INTERVAL_MS = 60000; // Update interval in milliseconds (60000ms = 1 minute)
const char* DEVICE_NAME = "ESP32-BME280";

// BME280 I2C pins (ESP32 DevKit V1)
#define SDA_PIN 21
#define SCL_PIN 22

// BLE UUIDs (Environmental Sensing Service standard)
#define SERVICE_UUID        "181A"  // Environmental Sensing Service
#define HUMIDITY_CHAR_UUID  "2A6F"  // Humidity Characteristic
#define COMMAND_CHAR_UUID   "00002A00-0000-1000-8000-00805f9b34fb"  // Custom command characteristic

// ===== GLOBAL VARIABLES =====
Adafruit_BME280 bme;
BLEServer* pServer = nullptr;
BLECharacteristic* pHumidityCharacteristic = nullptr;
BLECharacteristic* pCommandCharacteristic = nullptr;
bool deviceConnected = false;
unsigned long lastUpdateTime = 0;
bool triggerReading = false;  // Flag for on-demand readings

// ===== FORWARD DECLARATION =====
void sendHumidityReading();

// ===== BLE SERVER CALLBACKS =====
class ServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
        deviceConnected = true;
        Serial.println("BLE Client Connected!");
    }

    void onDisconnect(BLEServer* pServer) {
        deviceConnected = false;
        Serial.println("BLE Client Disconnected!");
        // Restart advertising
        pServer->startAdvertising();
        Serial.println("Advertising restarted");
    }
};

// ===== COMMAND CHARACTERISTIC CALLBACK =====
class CommandCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
        std::string value = pCharacteristic->getValue();
        
        if (value.length() > 0) {
            Serial.print("Command received: ");
            for (int i = 0; i < value.length(); i++) {
                Serial.print(value[i]);
            }
            Serial.println();
            
            // Set flag to trigger reading in main loop
            triggerReading = true;
        }
    }
};

// ===== FUNCTION TO READ AND SEND HUMIDITY =====
void sendHumidityReading() {
    if (!deviceConnected) {
        Serial.println("No device connected - skipping reading");
        return;
    }

    float humidity = bme.readHumidity();
    
    if (isnan(humidity)) {
        Serial.println("Failed to read from BME280 sensor!");
        return;
    }

    Serial.print("Humidity: ");
    Serial.print(humidity);
    Serial.println(" %");

    // Convert humidity to uint16 format (humidity * 100) as per BLE spec
    uint16_t humidityValue = (uint16_t)(humidity * 100);
    
    // Send as notification
    pHumidityCharacteristic->setValue(humidityValue);
    pHumidityCharacteristic->notify();
    
    Serial.println("Humidity reading sent via BLE");
}

// ===== SETUP =====
void setup() {
    Serial.begin(115200);
    Serial.println("\n=== ESP32 BME280 BLE Humidity Monitor ===");
    
    // Initialize I2C with custom pins
    Wire.begin(SDA_PIN, SCL_PIN);
    
    // Initialize BME280
    Serial.println("Initializing BME280...");
    if (!bme.begin(0x76)) {  // Default I2C address is 0x76, try 0x77 if this fails
        Serial.println("Could not find BME280 sensor! Check wiring.");
        Serial.println("Trying alternate address 0x77...");
        if (!bme.begin(0x77)) {
            Serial.println("BME280 initialization failed!");
            while (1) delay(10);
        }
    }
    Serial.println("BME280 initialized successfully!");
    
    // Configure BME280 for weather monitoring
    bme.setSampling(Adafruit_BME280::MODE_NORMAL,
                    Adafruit_BME280::SAMPLING_X1,  // temperature
                    Adafruit_BME280::SAMPLING_X1,  // pressure
                    Adafruit_BME280::SAMPLING_X1,  // humidity
                    Adafruit_BME280::FILTER_OFF);
    
    // Initialize BLE
    Serial.println("Initializing BLE...");
    BLEDevice::init(DEVICE_NAME);
    
    // Create BLE Server
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());
    
    // Create BLE Service
    BLEService *pService = pServer->createService(SERVICE_UUID);
    
    // Create Humidity Characteristic
    pHumidityCharacteristic = pService->createCharacteristic(
        HUMIDITY_CHAR_UUID,
        BLECharacteristic::PROPERTY_READ |
        BLECharacteristic::PROPERTY_NOTIFY
    );
    
    // Add descriptor for notifications
    pHumidityCharacteristic->addDescriptor(new BLE2902());
    
    // Add presentation format descriptor (tells apps how to display the value)
    BLEDescriptor *pFormatDescriptor = new BLEDescriptor(BLEUUID((uint16_t)0x2904));
    uint8_t formatData[7] = {
        0x06,  // Format: uint16
        0xFE,  // Exponent: -2 (divide by 100)
        0x27, 0x27,  // Unit: percentage (0x27AD in little-endian)
        0x01,  // Namespace: Bluetooth SIG
        0x00, 0x00  // Description: none
    };
    pFormatDescriptor->setValue(formatData, 7);
    pHumidityCharacteristic->addDescriptor(pFormatDescriptor);
    
    // Create Command Characteristic for on-demand readings
    pCommandCharacteristic = pService->createCharacteristic(
        COMMAND_CHAR_UUID,
        BLECharacteristic::PROPERTY_WRITE
    );
    pCommandCharacteristic->setCallbacks(new CommandCallbacks());
    
    // Start the service
    pService->start();
    
    // Start advertising
    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(0x06);
    pAdvertising->setMinPreferred(0x12);
    BLEDevice::startAdvertising();
    
    Serial.println("BLE Device is ready!");
    Serial.print("Device name: ");
    Serial.println(DEVICE_NAME);
    Serial.print("Update interval: ");
    Serial.print(UPDATE_INTERVAL_MS / 1000);
    Serial.println(" seconds");
    Serial.println("\nWaiting for LightBlue connection...");
    Serial.println("To get on-demand reading: Write any value to the Command characteristic");
}

// ===== LOOP =====
void loop() {
    // Check for on-demand reading trigger
    if (triggerReading) {
        sendHumidityReading();
        triggerReading = false;
    }
    
    // Send automatic updates at configured interval
    if (deviceConnected && (millis() - lastUpdateTime >= UPDATE_INTERVAL_MS)) {
        sendHumidityReading();
        lastUpdateTime = millis();
    }
    
    delay(100);  // Small delay to prevent excessive CPU usage
}