cat > download-weights-local.py << EOL
import shutil

source_folder = 'diffusers-cache2'
destination_folder = 'diffusers-cache'

shutil.copytree(source_folder, destination_folder)

EOL