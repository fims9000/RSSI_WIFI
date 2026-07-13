
#define DEBUG_WEBSOCKETS

#include <ESP8266WiFi.h>
#include <WebSocketsClient.h>

// const char* ssid = "AI";
// const char* password = "raogqm3e";

const char* ssid = "POCOF6";
const char* password = "123456789";
const char* websocket_host = "10.190.92.17";
const uint16_t websocket_port = 5000;

WebSocketsClient webSocket;

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("❌ WebSocket disconnected");
      break;
    case WStype_CONNECTED:
      Serial.println("✅ WebSocket connected");
      break;
    case WStype_TEXT:
      Serial.printf("📩 Message from server: %s\n", payload);
      break;
    default:
      break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(100);
  
  WiFi.begin(ssid, password);
  Serial.print("🔄 Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ Wi-Fi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  webSocket.begin(websocket_host, websocket_port, "/");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000); // пробовать reconnect каждые 5 сек
}

void loop() {
  webSocket.loop(); // обязательно для поддержания соединения

  static unsigned long lastSend = 0;
  // отправляем каждые 50 мс (20 Гц)
  if (millis() - lastSend >= 50) {
    lastSend = millis();
    
    if (webSocket.isConnected()) {
      int rssi = WiFi.RSSI();
      String data = String(millis()) + "," + String(rssi) + "\n";
      webSocket.sendTXT(data);
    } else {
      static unsigned long lastWarn = 0;
      if (millis() - lastWarn >= 5000) {
        Serial.println("⚠️ WebSocket not connected, waiting...");
        lastWarn = millis();
      }
    }
  }

  // небольшая задержка для предотвращения WDT (можно убрать, если всё стабильно)
  delay(1);
}