import csv

def create_default_embeddings_csv(num_patients=200, filename="embedding_csv.csv"):
    # T1-12 L1-5.
    bones = [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]
    
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Header.
        writer.writerow(["patch_id", "concept_label"])
        
        # 200 default patient number
        for patient_num in range(1, num_patients + 1):
            # patient_001, patient_002 format
            patient_id = f"patient_{patient_num:03d}"
            
            # 17 bones
            for bone in bones:
                #  Ex: patient_001_T1
                patch_id = f"{patient_id}_{bone}"
                
                # Default naming EX: T1 vertebra healthy
                concept_label = f"{bone} vertebra healthy"
                
                writer.writerow([patch_id, concept_label])
                
    print(f"Process done. {num_patients} patients , {num_patients * 17} line written to '{filename}'!")

if __name__ == "__main__":
    create_default_embeddings_csv()