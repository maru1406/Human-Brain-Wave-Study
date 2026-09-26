from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import Prediction
from django.db.models import Count

import numpy as np
from .ml_model import predict_emotion

@login_required
def dashboard_view(request):
    return render(request, 'dashboard/dashboard.html')

@login_required
def predict_view(request):

    result = None
    confidence_percent = None

    if request.method == "POST" and request.FILES.get("input_file"):

        uploaded_file = request.FILES["input_file"]

        sample = np.load(uploaded_file)

        predicted_class, confidence = predict_emotion(sample)

        confidence_percent = round(confidence * 100, 2)

        # Save prediction
        Prediction.objects.create(
            user=request.user,
            input_file=uploaded_file,
            predicted_class=predicted_class,
            confidence=confidence
        )

        result = predicted_class

    return render(request, "dashboard/predict.html", {
        "result": result,
        "confidence": confidence_percent
    })

@login_required
def history_view(request):
    qs = Prediction.objects.filter(user=request.user)

    # Count per class
    class_counts = qs.values('predicted_class').annotate(count=Count('id'))

    labels = [item['predicted_class'] for item in class_counts]
    data = [item['count'] for item in class_counts]

    return render(request, 'dashboard/history.html', {'labels': labels,'data': data,})

@login_required
def profile_page(request):
    profile = request.user.profile
    return render(request, 'dashboard/profile.html', {'profile': profile})

@login_required
def my_predictions(request):
    predictions = Prediction.objects.filter(user=request.user)
    return render(request, 'dashboard/my_predictions.html', {'predictions': predictions})
