import serial
import time
import re

ser = serial.Serial(
    port='COM9',
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1
)

print("Conectado. Enviando 'P' cada segundo...")
try:
    while True:
        ser.write(b'P')  # Solo 'P', sin \r
        print("Enviado 'P'")
        time.sleep(0.5)
        
        if ser.in_waiting:
            data = ser.read(ser.in_waiting)
            print("✅ Recibido:", data)
            try:
                text = data.decode('utf-8', errors='replace').strip()
                print("📄 Texto:", repr(text))
                nums = re.findall(r'\d+\.?\d*', text)
                if nums:
                    peso = float(nums[0])
                    print(f"⚖️ Peso detectado: {peso} {'g' if peso > 100 else 'kg'}")
            except Exception as e:
                print("❌ Decodificación fallida:", e)
        else:
            print("⏳ Sin respuesta.")
        
        time.sleep(1)

except KeyboardInterrupt:
    ser.close()