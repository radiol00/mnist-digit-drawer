from drawer import MonoQDigitDrawer
import torch
from classifier import MnistDigitClassifier

if __name__ == "__main__":
    
    classifier = MnistDigitClassifier()
    classifier.load_state_dict(torch.load("Classifier.pt"))
    drawer = MonoQDigitDrawer()
    
    def predict():
        img = drawer.get_image()
        img = torch.tensor(img.getdata(), dtype=torch.float) / 255
        preditions = classifier(img.view(1, -1)).squeeze()
        drawer.update_predictions(preditions)

    drawer.bind_predict_button(predict)
    drawer.show()


