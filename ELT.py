import os
from dotenv import load_dotenv
import synapseclient
from synapseclient import File

def download_all_files(syn, parent_id, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    children = list(syn.getChildren(parent_id))
    
    for item in children:
        item_name = item['name']
        item_id = item['id']
        item_type = item['type']
        local_path = os.path.join(local_dir, item_name)

        if item_type == 'org.sagebionetworks.repo.model.FileEntity':
            print(f"🔽 Descargando archivo: {item_name}")
            entity = syn.get(item_id, downloadLocation=local_dir)
            print("✅ Guardado en:", entity.path)
        elif item_type == 'org.sagebionetworks.repo.model.Folder':
            print(f"📂 Entrando en carpeta: {item_name}")
            download_all_files(syn, item_id, local_path)

# Cargar token
load_dotenv()
token = os.getenv("SYNAPSE_TOKEN")

# Conectar con Synapse
syn = synapseclient.Synapse()
syn.login(authToken=token)

# ID del proyecto
project_id = "syn60868042"

# Descargar todo
download_all_files(syn, project_id, "./datos")