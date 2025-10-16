from flask import Flask, jsonify, request
from flask_cors import CORS
import predict  # Imports your logic file
import logging

# --- INITIAL SETUP ---
app = Flask(__name__)
CORS(app) 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API ROUTES ---

@app.route("/api/materials", methods=["GET"])
def get_materials_route():
    """Fetches the list of materials."""
    try:
        materials = predict.get_materials()
        return jsonify(materials)
    except Exception as e:
        logging.error(f"Error in /api/materials: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["GET"])
def predict_price_route():
    """Gets price prediction for a material."""
    material_id = request.args.get("material_id")
    if not material_id:
        return jsonify({"error": "material_id is required"}), 400
    try:
        historical, predictions = predict.get_price_prediction(material_id)
        if historical is None:
            return jsonify({"error": f"Not enough data to predict for '{material_id}'"}), 404
        return jsonify({
            "material_id": material_id,
            "historical_data": historical,
            "predictions": predictions
        })
    except Exception as e:
        logging.error(f"Error in /api/predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/recommend_vendor", methods=["GET"])
def recommend_vendor_route():
    """Gets vendor recommendation with dynamic weights."""
    material_id = request.args.get("material_id")
    if not material_id:
        return jsonify({"error": "material_id is required"}), 400
    
    weights = {
        'price': float(request.args.get('w_price', 20)) / 100.0,
        'delivery': float(request.args.get('w_delivery', 30)) / 100.0,
        'reliability': float(request.args.get('w_reliability', 50)) / 100.0
    }
        
    try:
        best_vendor, all_vendors, score_breakdown = predict.get_vendor_recommendation(material_id, weights)
        if best_vendor is None:
            return jsonify({"error": f"No vendors found for material '{material_id}'"}), 404
        return jsonify({
            "material_id": material_id,
            "best_vendor": best_vendor,
            "all_vendors": all_vendors,
            "weighted_score": best_vendor.get('final_score', 0),
            "score_breakdown": score_breakdown
        })
    except Exception as e:
        logging.error(f"Error in /api/recommend_vendor: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/add_requirement", methods=["POST"])
def add_requirement_route():
    """Receives a procurement requirement and checks for feasible vendors."""
    data = request.get_json()
    if not data or not all(k in data for k in ["material_id", "quantity", "deadline"]):
        return jsonify({"error": "Missing fields: material_id, quantity, deadline"}), 400
    
    try:
        result = predict.check_requirement(data)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in /api/add_requirement: {e}")
        return jsonify({"error": str(e)}), 500
        
# --- SERVER STARTUP ---
if __name__ == "__main__":
    logging.info("Starting Flask server...")
    predict.initialize_data_from_csv() # Make sure data is loaded
    app.run(host='0.0.0.0', debug=True, port=5000)

