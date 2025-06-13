import os
from dotenv import load_dotenv
import synapseclient
from synapseclient import File

def download_all_files(syn, parent_id, local_dir):
    """Descarga recursivamente todos los archivos de un proyecto Synapse"""
    os.makedirs(local_dir, exist_ok=True)
    children = list(syn.getChildren(parent_id))
    
    for item in children:
        item_name = item['name']
        item_id = item['id']
        item_type = item['type']
        local_path = os.path.join(local_dir, item_name)
        
        if item_type == 'org.sagebionetworks.repo.model.FileEntity':
            print(f"🔽 Descargando archivo: {item_name}")
            try:
                entity = syn.get(item_id, downloadLocation=local_dir)
                print(f"✅ Guardado en: {entity.path}")
            except Exception as e:
                print(f"❌ Error descargando {item_name}: {e}")
                
        elif item_type == 'org.sagebionetworks.repo.model.Folder':
            print(f"📂 Entrando en carpeta: {item_name}")
            download_all_files(syn, item_id, local_path)

def main():
    # Cargar token desde .env
    load_dotenv()
    token = os.getenv("SYNAPSE_TOKEN")
    
    if not token:
        print("❌ Error: SYNAPSE_TOKEN no encontrado en .env")
        return
    
    try:
        # Conectar con Synapse
        print("🔗 Conectando con Synapse...")
        syn = synapseclient.Synapse()
        syn.login(authToken=token)
        print("✅ Conectado exitosamente")
        
        # ID de los pesos de MAMA-MIA (Dataset101)
        weights_id = "syn61247992"
        local_dir = "./mama_mia_weights"
        
        print(f"📥 Descargando pesos de MAMA-MIA desde {weights_id}...")
        download_all_files(syn, weights_id, local_dir)
        
        print("\n🎉 ¡Descarga completada!")
        print(f"📂 Archivos guardados en: {local_dir}")
        
        # Listar archivos descargados
        if os.path.exists(local_dir):
            print("\n📋 Archivos descargados:")
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                    print(f"  📄 {file} ({file_size:.1f} MB)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()