import os
import json
import glob

output_dir = "/data/wzx/output/best_result"
json_files = glob.glob(os.path.join(output_dir, "result_*.json"))

for json_file in sorted(json_files):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        with open("dumped_text.txt", "a", encoding="utf-8") as out:
            out.write(f"FILE: {os.path.basename(json_file)}\n")
            out.write(f"PATIENT: {data.get('input_description', '')}\n")
            out.write(f"INTERVENTION: {data.get('intervention', '')}\n")
            
            # Try to find reasoning in the top level or nested
            reasoning = data.get('reasoning', '')
            if not reasoning:
                 if 'prediction_data' in data and 'prediction' in data['prediction_data']:
                     reasoning = data['prediction_data']['prediction'].get('intervention_analysis', {}).get('reasoning', '')
            
            out.write(f"REASONING: {reasoning}\n")
            out.write(f"RISK: {data.get('risk_level', '')}\n")
            out.write("-" * 40 + "\n")
        
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
