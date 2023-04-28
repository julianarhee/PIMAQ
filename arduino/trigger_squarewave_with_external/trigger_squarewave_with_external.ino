#include <TimerOne.h>
#include <SoftwareSerial.h>
// check signal parameters from serial
// if first charakter 'S' start interrupt
// integer following S ist the FPS-setting
// if first charakter 'Q' stop interrupt
// dutycycle is set in dependancy of freq  to be ~ 5ms
// if first charakter 'T' send following string to intan
const byte rxPin = 2;
const byte txPin = 3;
SoftwareSerial mySerial(rxPin, txPin, 1);
const int LedPin = 9;
const int external_trigger = 8;
bool external_status = 0;
bool pinStatus = 0;
long fps = 120;
float duration = 5000.0;
int input;
float dutyCycle = 30.0;

void setup(void)
{
  Serial.begin(115200); //9600);
  //mySerial.begin(600);
  pinMode(external_trigger, INPUT);
  pinMode(LedPin, OUTPUT);
  digitalWrite(LedPin, LOW);
  delay(100);
}

void loop(void)
{
  if (digitalRead(external_trigger)==HIGH){
    external_status = 1;
  }
  else{
    external_status = 0;
  }
  if ((pinStatus == 1) && (external_status==0)){
    Serial.println(String('Q'));
  }
    
  if (Serial.available() > 0)
  {
    input = Serial.read();
    if (input == 'P') // Poll the arduino, expect answer bit '1'
    {
      Serial.print(1);
    }
    if (input == 'Q')
    {
      if (pinStatus == 1)
      {
        Timer1.disablePwm(LedPin);
        Timer1.stop();
        Serial.println(3);
        pinStatus = 0;
        external_status = 0;
        //digitalWrite(external_trigger, LOW);
      }
      external_status = 0;
      //Serial.println(digitalRead(external_trigger));
    }

    if (input == 'S')
    {
      fps = Serial.parseInt();
      Serial.println(2);
      Serial.print(fps);
      Serial.println(digitalRead(external_trigger)); //digitalRead(external_trigger));

      //duration=(float)Serial.parseInt(); // if we also want to set a duration of pulse
      while((external_status == 0) && (pinStatus==0)){
        // just wait
        if (digitalRead(external_trigger)==HIGH){
          external_status = 1;
        }
      }
      if ((pinStatus == 0) && (external_status==1))
      {
        Timer1.initialize(1000000 / fps); // 40 us = 25 kHz
        dutyCycle = duration / (1000000 / fps); // calculate duty cycle to have pulse length ~5ms
        Timer1.pwm(LedPin, (dutyCycle) * 1023);
        pinStatus = 1;
      }
    if (input == 'T')
    {
      String text = Serial.readString();
      mySerial.print(text); //Write the text from Serial port
    }
    delay(100);
  }
}
}
