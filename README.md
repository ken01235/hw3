# hw3 108061207
The main.cpp file is in /src/model_deploy/, and use "sudo mbed compile --source . --source ~/ee2405/mbed-os-build/ -m B_L4S5I_IOT01A -t GCC_ARM --profile tflite.json -f" to compile it.
In the screen, type "/gesture/run" to get in gesture mode. In gesture mode, uLCD will display the gesture ID. Rotating the mbed counterclockwisely is 90 degree. Rotating the mbed clockwisely is 45 degree. The slope motion from Lab8 is degree. Then press the user button to selection one angle as the threshold angle. In the meanwhile, mbed board will send the angle to broker.
In the screen, type "/tilt/run" to get in tilt detecting mode. If the tilt angle is over the selected threshold angle, mbed will publish the event and angle through WiFi/MQTT to a broker and quit the tilt detecting mode.
