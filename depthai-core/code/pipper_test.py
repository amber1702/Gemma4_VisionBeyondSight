import subprocess

text = "Có người phía trước"

cmd = f'''
echo "{text}" | \
piper \
--model piper_models/vi_VN-vais1000-medium.onnx \
--output_file out.wav
'''

subprocess.run(cmd, shell=True)

subprocess.run("aplay out.wav", shell=True)
