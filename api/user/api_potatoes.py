from api.user.serializers import UserSerializer
from api.user.models import User
from rest_framework import viewsets, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.exceptions import ValidationError
from rest_framework import mixins
import numpy as np
import cv2
import os 
from rest_framework.views import APIView
from django.conf import settings
import tensorflow as tf
from django.http import JsonResponse
from io import BytesIO
from PIL import Image



class UserViewSet(
    viewsets.GenericViewSet, mixins.CreateModelMixin, mixins.UpdateModelMixin
):
    serializer_class = UserSerializer
    permission_classes = (IsAuthenticated,)

    error_message = {"success": False, "msg": "Error updating user"}

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop("partial", True)
        instance = User.objects.get(id=request.data.get("userID"))
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        if getattr(instance, "_prefetched_objects_cache", None):
            instance._prefetched_objects_cache = {}

        return Response(serializer.data)

    def create(self, request, *args, **kwargs):
        user_id = request.data.get("userID")

        if not user_id:
            raise ValidationError(self.error_message)

        if self.request.user.pk != int(user_id) and not self.request.user.is_superuser:
            raise ValidationError(self.error_message)

        self.update(request)

        return Response({"success": True}, status.HTTP_200_OK)



def load_model():
    model_path = os.path.join(settings.BASE_DIR, 'api','user', 'potatoes_model')
    return tf.keras.models.load_model(model_path)
def read_file_as_image(file):
    data = file.read()
    image = np.array(Image.open(BytesIO(data)).resize((256, 256)))
    return image
class ImageViewPotato(APIView):
    # @csrf_exempt
    def post(self, request):
        if request.method == 'POST':
            image_file = request.FILES.get('image')
            if image_file:
                # Récupérer l'image à partir du formulaire
                image = read_file_as_image(image_file)
                img_batch = np.expand_dims(image, 0)

                # Faire la prédiction
                model = load_model()
                CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
                predictions = model.predict(img_batch)
                predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
                confidence = np.max(predictions[0])
                attributes = {
                    "class": predicted_class,#test(predictions)
                    "confidence": float(confidence)
                }

            return JsonResponse(attributes)

        return JsonResponse({'error': 'Image upload failed'})

    def get(self,request):
        if request.method =="GET":
            print("predicting plant classname...")
        return JsonResponse({'error': 'Invalid request method'})