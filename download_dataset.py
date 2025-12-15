from datasets import load_dataset
import json
from pathlib import Path
import random

print("🌾 Creating Agricultural Expert Dataset...\n")

formatted = []

# System prompt for agricultural assistant
SYSTEM = """You are an Agricultural Expert Assistant. Help farmers with:
1. Crop disease identification and treatment
2. Pest identification and control
3. Fertilizer recommendations
4. Planting schedules and crop rotation
Provide practical, actionable advice for farmers."""

# Download agricultural Q&A datasets
print("1️⃣ Downloading PlantVillage dataset...")
try:
    # PlantVillage dataset (crop diseases)
    ds = load_dataset("gaborcselle/plant-village", split="train[:500]")
    print(f"   Downloaded {len(ds)} plant disease examples")
except Exception as e:
    print(f"   PlantVillage not available: {e}")

# Create agricultural examples
print("\n2️⃣ Creating agricultural expert examples...")

agro_examples = [
    {
        "q": "My tomato leaves have yellow spots and are curling. What could be the problem?",
        "a": "**DIAGNOSIS:** Likely Tomato Yellow Leaf Curl Virus (TYLCV) or Early Blight\n\n**SYMPTOMS OBSERVED:**\n• Yellow spots on leaves\n• Leaf curling\n\n**TREATMENT:**\n• Remove affected leaves immediately\n• Apply neem oil spray (2-3ml/liter) weekly\n• Use copper-based fungicide if fungal\n• Ensure proper spacing for air circulation\n\n**PREVENTION:**\n• Use resistant varieties\n• Control whitefly population (virus vectors)\n• Maintain field hygiene"
    },
    {
        "q": "What fertilizer should I use for wheat in sandy soil?",
        "a": "**FERTILIZER RECOMMENDATION FOR WHEAT (Sandy Soil):**\n\n**NPK Ratio:** 120:60:40 kg/hectare\n\n**Application Schedule:**\n• Basal dose: 60:60:40 kg NPK at sowing\n• First top dressing: 30 kg N at 21 days\n• Second top dressing: 30 kg N at 45 days\n\n**Sandy Soil Special Care:**\n• Add organic matter (10-15 tons/hectare)\n• Apply fertilizers in splits to prevent leaching\n• Consider slow-release formulations\n• Increase potassium by 20% for better water retention\n\n**Micronutrients:**\n• Zinc sulfate: 25 kg/hectare\n• Boron: 1 kg/hectare if deficient"
    },
    {
        "q": "White insects on my cotton plants, leaves turning yellow",
        "a": "**PEST IDENTIFIED:** Whitefly (Bemisia tabaci)\n\n**DAMAGE SYMPTOMS:**\n• Yellowing leaves (chlorosis)\n• Sticky honeydew on leaves\n• Sooty mold development\n• Leaf curl in severe cases\n\n**IMMEDIATE CONTROL:**\n• Yellow sticky traps: 40-50/hectare\n• Spray: Imidacloprid 17.8% SL @ 0.3ml/liter\n• Alternative: Thiamethoxam 25% WG @ 0.4g/liter\n\n**ORGANIC OPTIONS:**\n• Neem oil 3000ppm @ 5ml/liter\n• Verticillium lecanii @ 5g/liter\n\n**PREVENTION:**\n• Remove weeds (alternate hosts)\n• Avoid excessive nitrogen\n• Maintain natural predators"
    },
    {
        "q": "Best time to plant rice in tropical climate?",
        "a": "**RICE PLANTING SCHEDULE (Tropical Climate):**\n\n**MAIN SEASONS:**\n\n🌧️ **Wet Season (Main Crop):**\n• Sowing: May-June\n• Transplanting: June-July\n• Harvest: October-November\n• Advantage: Natural rainfall, less irrigation\n\n☀️ **Dry Season (Second Crop):**\n• Sowing: December-January\n• Transplanting: January-February\n• Harvest: April-May\n• Requires: Assured irrigation\n\n**VARIETY SELECTION:**\n• Wet season: Long duration (140-150 days)\n• Dry season: Short duration (110-120 days)\n\n**KEY FACTORS:**\n• Temperature: 20-35°C optimal\n• Avoid flowering during extreme heat\n• Plan 15-20 days for nursery"
    },
    {
        "q": "Brown spots on potato leaves spreading fast",
        "a": "**DISEASE:** Late Blight (Phytophthora infestans)\n\n**IDENTIFICATION:**\n• Brown/black spots with yellow halo\n• White fungal growth on leaf underside\n• Rapid spread in humid conditions\n• Can destroy crop in 7-10 days\n\n**IMMEDIATE ACTION:**\n• Spray Metalaxyl + Mancozeb @ 2.5g/liter\n• Alternative: Copper oxychloride @ 3g/liter\n• Repeat every 7-10 days\n• Stop irrigation temporarily\n\n**PREVENTION:**\n• Plant resistant varieties\n• Proper drainage\n• Destroy infected plant debris\n• Crop rotation with non-solanaceous crops"
    },
    {
        "q": "My corn has holes in leaves and stem, what pest is this?",
        "a": "**PEST:** Fall Armyworm (Spodoptera frugiperda) or Corn Borer\n\n**DAMAGE IDENTIFICATION:**\n• Pin holes in leaves\n• Windowpane feeding pattern\n• Bore holes in stem\n• Frass (insect droppings) visible\n\n**CONTROL MEASURES:**\n• Spray: Emamectin benzoate 5% SG @ 0.4g/liter\n• Or: Chlorantraniliprole 18.5% SC @ 0.3ml/liter\n• Apply in evening when larvae are active\n\n**BIOLOGICAL CONTROL:**\n• Trichogramma cards: 50,000 eggs/hectare\n• Bacillus thuringiensis @ 2ml/liter\n\n**CULTURAL PRACTICES:**\n• Deep plowing after harvest\n• Intercrop with legumes\n• Bird perches for natural predation"
    },
    {
        "q": "How much water does sugarcane need?",
        "a": "**SUGARCANE WATER REQUIREMENTS:**\n\n**Total Water Need:** 1800-2500mm per crop cycle\n\n**STAGE-WISE IRRIGATION:**\n\n🌱 **Germination (0-30 days):**\n• Light irrigation every 3-4 days\n• 50-60mm total\n\n🌿 **Tillering (30-120 days):**\n• Every 7-10 days\n• 400-500mm total\n\n📈 **Grand Growth (120-270 days):**\n• Maximum water need\n• Every 7 days\n• 1000-1200mm total\n\n🌾 **Maturity (270-360 days):**\n• Reduce irrigation\n• Every 15-20 days\n• 200-300mm total\n• Stop 15 days before harvest\n\n**IRRIGATION METHODS:**\n• Drip: 40% water saving\n• Furrow: Traditional\n• Sprinkler: For light soils"
    },
    {
        "q": "Organic fertilizer options for vegetables",
        "a": "**ORGANIC FERTILIZERS FOR VEGETABLES:**\n\n**COMPOST TYPES & APPLICATION:**\n\n🌱 **Farmyard Manure (FYM):**\n• Rate: 20-25 tons/hectare\n• NPK: 0.5:0.2:0.5%\n• Apply 2 weeks before planting\n\n🐓 **Poultry Manure:**\n• Rate: 5-8 tons/hectare\n• NPK: 3:2:2%\n• Very concentrated, use carefully\n\n🌿 **Vermicompost:**\n• Rate: 5-7 tons/hectare\n• NPK: 1.5:0.5:1.5%\n• Excellent for seedlings\n\n**GREEN MANURES:**\n• Sunhemp: 25-30 kg N/hectare\n• Dhaincha: 20-25 kg N/hectare\n• Grow 45-60 days, incorporate\n\n**LIQUID ORGANICS:**\n• Panchagavya: 3% spray\n• Fish emulsion: 1:100 dilution\n• Seaweed extract: 0.5ml/liter\n\n**APPLICATION TIPS:**\n• Leafy vegetables: High nitrogen\n• Root vegetables: High potassium\n• Fruiting vegetables: Balanced NPK"
    },
    {
        "q": "Yellowing between leaf veins in citrus trees",
        "a": "**PROBLEM:** Iron Deficiency (Chlorosis)\n\n**SYMPTOMS:**\n• Interveinal chlorosis (yellowing between veins)\n• Veins remain green\n• Young leaves affected first\n• Reduced fruit size\n\n**IMMEDIATE TREATMENT:**\n• Foliar spray: Ferrous sulfate 0.5% + Lime 0.25%\n• Or: Chelated iron (Fe-EDTA) @ 10g/tree\n• Spray early morning or evening\n\n**SOIL APPLICATION:**\n• Ferrous sulfate: 100-200g/tree\n• Mix with organic matter\n• Apply in basin around tree\n\n**LONG-TERM CORRECTION:**\n• Lower soil pH with sulfur\n• Add organic matter\n• Improve drainage\n• Avoid excess phosphorus\n\n**PREVENTION:**\n• Regular soil testing\n• Use iron-efficient rootstocks\n• Mulching to maintain soil moisture"
    },
    {
        "q": "Rats damaging my rice field at grain filling stage",
        "a": "**RODENT CONTROL IN RICE:**\n\n**DAMAGE ASSESSMENT:**\n• Cut tillers at base\n• Grain eating at milk/dough stage\n• Burrows on bunds\n• Economic threshold: 2-3 active burrows/100m²\n\n**INTEGRATED MANAGEMENT:**\n\n**Physical Control:**\n• Community trapping campaign\n• Burrow smoking\n• Flood burrows during land preparation\n\n**Chemical Control:**\n• Zinc phosphide bait @ 2%\n• Bromadiolone cakes in burrows\n• Place in evening, collect dead rats morning\n\n**BIOLOGICAL:**\n• Owl perches (1/hectare)\n• Protect natural predators\n\n**CULTURAL:**\n• Synchronous planting\n• Clean bunds and surroundings\n• Remove weeds (hiding places)\n• Community-wide action essential\n\n**TIMING:** Most effective 2-3 weeks after transplanting"
    },
    {
        "q": "Best crop rotation for soil health after growing cotton",
        "a": "**CROP ROTATION AFTER COTTON:**\n\n**WHY ROTATION NEEDED:**\n• Cotton depletes soil nutrients\n• Pest/disease buildup\n• Soil structure degradation\n\n**RECOMMENDED SEQUENCES:**\n\n**Option 1 (Best for soil):**\nCotton → Legumes (Green gram/Black gram) → Wheat → Fallow/Green manure\n\n**Option 2 (Profitable):**\nCotton → Maize + Legume intercrop → Wheat → Soybean\n\n**Option 3 (Water-saving):**\nCotton → Chickpea → Sorghum → Groundnut\n\n**BENEFITS BY CROP:**\n• Legumes: Fix 40-60 kg N/hectare\n• Cereals: Different nutrient needs\n• Deep-rooted crops: Break hardpan\n\n**SOIL IMPROVEMENT:**\n• Add 10-15 tons FYM/hectare\n• Grow dhaincha in summer\n• Retain crop residues\n• Minimum tillage where possible"
    },
    {
        "q": "Black sooty coating on mango leaves",
        "a": "**PROBLEM:** Sooty Mold (Secondary infection)\n\n**PRIMARY CAUSE:**\n• Honeydew from sucking pests\n• Usually mealybugs, aphids, or scale insects\n\n**MANAGEMENT STRATEGY:**\n\n**Step 1 - Control Primary Pests:**\n• Spray: Imidacloprid @ 0.5ml/liter\n• Or: Fish oil rosin soap @ 25g/liter\n• Target insects producing honeydew\n\n**Step 2 - Remove Sooty Mold:**\n• Spray starch solution (1%)\n• Let dry and wash with water\n• Or use soap solution spray\n\n**Step 3 - Preventive Spray:**\n• Neem oil @ 5ml/liter monthly\n• Maintains pest control\n\n**CULTURAL PRACTICES:**\n• Prune for better air circulation\n• Remove ant colonies (they farm aphids)\n• Maintain tree vigor with nutrition"
    },
    {
        "q": "Soil test shows pH 8.5, how to reduce it?",
        "a": "**ALKALINE SOIL CORRECTION (pH 8.5):**\n\n**AMENDMENTS TO LOWER pH:**\n\n**Elemental Sulfur (Most effective):**\n• Sandy soil: 400-500 kg/hectare\n• Clay soil: 800-1000 kg/hectare\n• Takes 6-12 months for full effect\n\n**Sulfuric Acid:**\n• Through irrigation: 0.1-0.2%\n• Faster but requires careful handling\n\n**Organic Matter:**\n• FYM: 20-30 tons/hectare\n• Compost: Creates organic acids\n• Press mud: 10 tons/hectare\n\n**GYPSUM (Calcium Sulfate):**\n• Rate: 3-5 tons/hectare\n• Doesn't lower pH but improves soil\n• Helps with sodium problems\n\n**MANAGEMENT PRACTICES:**\n• Grow green manures (dhaincha)\n• Use acidifying fertilizers (ammonium sulfate)\n• Avoid irrigation with alkaline water\n• Apply amendments in splits\n• Retest soil after 6 months\n\n**TARGET:** Bring pH to 6.5-7.5 gradually"
    },
    {
        "q": "When should I harvest onions and how to store them?",
        "a": "**ONION HARVESTING & STORAGE:**\n\n**HARVEST MATURITY SIGNS:**\n• 50-70% tops fall over naturally\n• Neck becomes soft\n• Outer scales dry and papery\n• 120-150 days from transplanting\n\n**HARVESTING METHOD:**\n• Stop irrigation 10-15 days before\n• Harvest in dry weather\n• Pull/dig carefully to avoid bruising\n• Leave in field 3-5 days for curing\n\n**FIELD CURING:**\n• Place in windrows\n• Cover bulbs with tops\n• Protects from sunburn\n• Continue until necks dry\n\n**STORAGE PREPARATION:**\n• Remove tops leaving 2-3cm neck\n• Grade by size\n• Remove damaged/diseased bulbs\n\n**STORAGE CONDITIONS:**\n• Temperature: 25-30°C\n• Humidity: 65-70%\n• Good ventilation essential\n• Stack in mesh bags or crates\n• Can store 4-6 months\n\n**LOSSES PREVENTION:**\n• Avoid storage of thick-neck bulbs\n• Regular inspection\n• Remove sprouted/rotted bulbs"
    },
    {
        "q": "My banana plants leaves are turning yellow from bottom",
        "a": "**DIAGNOSIS:** Likely Nitrogen Deficiency or Panama Disease\n\n**IF NITROGEN DEFICIENCY:**\n• Older leaves yellow first\n• Uniform yellowing\n• Stunted growth\n\n**Treatment:**\n• Urea: 200g/plant immediately\n• Follow with 100g monthly\n• Or DAP: 150g/plant\n\n**IF PANAMA DISEASE (Fusarium Wilt):**\n• Yellowing starts from margins\n• Brown discoloration in pseudostem\n• Progressive wilting\n\n**Management:**\n• No cure - remove affected plants\n• Burn infected material\n• Apply lime to soil\n• Plant resistant varieties (Grand Naine)\n• Crop rotation for 3-4 years\n\n**DIFFERENTIATION TEST:**\n• Cut pseudostem cross-section\n• Brown/black = Disease\n• Clear = Nutrient issue\n\n**PREVENTION:**\n• Good drainage\n• Balanced nutrition\n• Use disease-free suckers"
    }
]

# Add examples multiple times for better training
for _ in range(3):  # Repeat 3 times
    for ex in agro_examples:
        formatted.append({
            "instruction": f"{SYSTEM}\n\nFarmer's question: {ex['q']}",
            "input": "",
            "output": ex['a']
        })

        # Also add without system prompt
        formatted.append({
            "instruction": ex['q'],
            "input": "",
            "output": ex['a']
        })

print(f"   ✅ Created {len(formatted)} agricultural examples\n")

# Try to get some real agricultural data
print("3️⃣ Attempting to download agricultural datasets...")
try:
    # Kisan Call Center dataset (Indian agricultural Q&A)
    ds = load_dataset("codefire007/Kisan-call-dataset", split="train[:200]")
    for item in ds:
        query = item.get("Query", "")
        response = item.get("Response", "")
        if query and response:
            formatted.append({
                "instruction": f"{SYSTEM}\n\nFarmer's question: {query}",
                "input": "",
                "output": response
            })
    print(f"   ✅ Added Kisan Call Center data\n")
except:
    print("   ℹ️ Kisan dataset not available\n")

print(f"📊 TOTAL: {len(formatted)} training examples")

# Shuffle for better training
random.shuffle(formatted)

# Save
Path("data").mkdir(exist_ok=True)
with open("data/agro_train.jsonl", "w", encoding="utf-8") as f:
    for item in formatted:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Saved to data/agro_train.jsonl")

# Save sample for inspection
with open("data/agro_sample.json", "w", encoding="utf-8") as f:
    json.dump(formatted[:3], f, indent=2, ensure_ascii=False)

print("✅ Sample saved to data/agro_sample.json")
print("\n🌾 Agricultural dataset ready for training!")