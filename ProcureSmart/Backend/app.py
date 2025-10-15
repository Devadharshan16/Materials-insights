from flask import Flask, jsonify, request
from flask_cors import CORS
import predict

app = Flask(__name__)
CORS(app) 

@app.route("/api/materials", methods=["GET"])
def get_materials_route():
    try:
        return jsonify(predict.get_materials())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["GET"])
def predict_price_route():
    material_id = request.args.get("material_id")
    if not material_id:
        return jsonify({"error": "material_id is required"}), 400
    try:
        historical, predictions = predict.get_price_prediction(material_id)
        if historical is None:
            return jsonify({"error": f"No data for material '{material_id}'"}), 404
        return jsonify({
            "material_id": material_id,
            "historical_data": historical,
            "predictions": predictions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/recommend_vendor", methods=["GET"])
def recommend_vendor_route():
    material_id = request.args.get("material_id")
    if not material_id:
        return jsonify({"error": "material_id is required"}), 400
    
    weights = {
        'price': float(request.args.get('w_price', 20)) / 100.0,
        'delivery': float(request.args.get('w_delivery', 30)) / 100.0,
        'reliability': float(request.args.get('w_reliability', 50)) / 100.0
    }
        
    try:
        best, all, breakdown = predict.get_vendor_recommendation(material_id, weights)
        if best is None:
            return jsonify({"error": f"No vendors for material '{material_id}'"}), 404
        return jsonify({
            "material_id": material_id, "best_vendor": best, "all_vendors": all,
            "weighted_score": best.get('final_score', 0), "score_breakdown": breakdown
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
