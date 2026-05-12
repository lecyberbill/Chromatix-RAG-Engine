import sys
import os

print("1. Ajout de Chromatix au path...")
sys.path.append(r"d:\image_to_text\chromatix")

try:
    print("2. Importation de CPSPacket...")
    from chromatix_cps.core import CPSPacket
    print("   CPSPacket... OK")
except Exception as e:
    print(f"   ERREUR CPSPacket: {e}")

try:
    print("3. Importation de TransformersHook...")
    from chromatix_cps.integrations import TransformersHook
    print("   TransformersHook... OK")
except Exception as e:
    print(f"   ERREUR TransformersHook: {e}")

print("4. Test terminé.")
