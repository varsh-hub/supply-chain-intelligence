def generate_recommendation(
    risk_level,
    risk_probability,
    anomaly_label,
    shipping_mode,
    scheduled_shipping_days,
    order_quantity,
    order_profit
):
    """
    Generate an explainable business recommendation
    based on risk and operational signals.
    """

    recommendations = []

    # --------------------------------------------------
    # 1. Overall risk
    # --------------------------------------------------

    if risk_level == "HIGH":

        recommendations.append(
            "Prioritize this order for immediate logistics review."
        )

    elif risk_level == "MEDIUM":

        recommendations.append(
            "Monitor this order closely and review shipment progress."
        )

    else:

        recommendations.append(
            "No immediate intervention is required; continue normal monitoring."
        )

    # --------------------------------------------------
    # 2. Anomaly detection
    # --------------------------------------------------

    if anomaly_label == 1:

        recommendations.append(
            "Investigate the unusual order pattern identified by "
            "the anomaly-detection model."
        )

    # --------------------------------------------------
    # 3. Shipping mode
    # --------------------------------------------------

    if shipping_mode == "First Class":

        recommendations.append(
            "Review First Class shipment performance because this "
            "shipping mode shows elevated late-delivery risk in the dataset."
        )

    elif shipping_mode == "Second Class":

        recommendations.append(
            "Monitor Second Class shipments because this shipping "
            "mode also shows elevated late-delivery risk."
        )

    # --------------------------------------------------
    # 4. Scheduled shipping duration
    # --------------------------------------------------

    if scheduled_shipping_days <= 2:

        recommendations.append(
            "Use proactive shipment monitoring because the scheduled "
            "shipping window is short."
        )

    # --------------------------------------------------
    # 5. Order quantity
    # --------------------------------------------------

    if order_quantity >= 5:

        recommendations.append(
            "Review the order quantity and confirm that fulfillment "
            "capacity is sufficient."
        )

    # --------------------------------------------------
    # 6. Profitability
    # --------------------------------------------------

    if order_profit < 0:

        recommendations.append(
            "Review the order's negative profitability and investigate "
            "whether shipping or fulfillment costs are contributing."
        )

    # --------------------------------------------------
    # Final recommendation
    # --------------------------------------------------

    return recommendations


def format_recommendations(
    recommendations
):
    """
    Convert recommendation list into a readable string.
    """

    return " ".join(
        recommendations
    )


if __name__ == "__main__":

    print("Testing recommendation engine...")

    recommendations = generate_recommendation(
        risk_level="HIGH",
        risk_probability=0.91,
        anomaly_label=1,
        shipping_mode="First Class",
        scheduled_shipping_days=2,
        order_quantity=6,
        order_profit=-500
    )

    print("\n========== RECOMMENDATIONS ==========")

    for recommendation in recommendations:

        print(f"- {recommendation}")