import reverse_geocoder as rg
import os

def translate_coords(lat, lon):
    try:
        # Feed the coordinates into the offline geocoder
        location = rg.search((lat, lon), verbose=False)
        city = location[0].get('name', 'Unknown')
        cc = location[0].get('cc', 'Unknown')
        return f"{city}, {cc}"
    except:
        return "Unknown Location"

def find_best_guesses():
    filepath = "guess_log.txt" 
    
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}")
        return

    guesses = []
    current_step = ""
    current_actual = ""
    
    print("Scanning logs and translating coordinates... (This takes a few seconds)")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("--- Step"):
                parts = line.split('|')
                current_step = parts[0].replace("---", "").strip()
                current_actual = parts[1].replace("---", "").strip().replace("Actual:", "").strip()
            
            # Find the #1 guess line
            if line.strip().startswith("1.") and 'Guess:' in line:
                try:
                    # Extract the distance
                    dist_str = line.split('|')[1].replace('km', '').strip()
                    dist = float(dist_str)
                    
                    # Extract the coordinates
                    # Looks like: Guess: ( 33.75, -117.98)
                    coord_str = line.split('Guess:')[1].strip().replace('(', '').replace(')', '')
                    lat_str, lon_str = coord_str.split(',')
                    lat, lon = float(lat_str), float(lon_str)
                    
                    guesses.append({
                        "step": current_step,
                        "actual": current_actual,
                        "distance": dist,
                        "lat": lat,
                        "lon": lon
                    })
                except Exception as e:
                    continue

    # Sort by distance (lowest error first)
    guesses.sort(key=lambda x: x["distance"])
    
    print("\n=== TOP 5 SUCCESS CASES ===")
    for g in guesses[:5]:
        pred_place = translate_coords(g['lat'], g['lon'])
        print(f"{g['step']} | Error: {g['distance']:>6.0f} km | Actual: {g['actual']:<15} | Predicted: {pred_place}")
        
    print("\n=== TOP 2 'SMART FAILS' (600km - 1000km error) ===")
    smart_fails = [g for g in guesses if 600 <= g["distance"] <= 1000]
    for g in smart_fails[:2]:
        pred_place = translate_coords(g['lat'], g['lon'])
        print(f"{g['step']} | Error: {g['distance']:>6.0f} km | Actual: {g['actual']:<15} | Predicted: {pred_place}")

if __name__ == "__main__":
    find_best_guesses()