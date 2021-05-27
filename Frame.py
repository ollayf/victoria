from utils.annotation import draw_bbox
import cv2 
class Frame:
    '''
    Class used to store information about the frame
    '''
    image = None
    annotated_img = None
    desired = 0
    desired_name = 0
    min_score = 0.28

    def __init__(self, im_array, desired_classes: list= [1], desired:int = 2 ,desired_name='Person') -> None:
        '''
        @params desired: 0 for undesired only, 2 for desired only, 1 for both
        '''
        self.image = self._prepare_img(im_array)
        self.desired_classes = desired_classes
        self.desired = desired
        self.desired_name = desired_name
        self.des_predictions = []
        self.undes_predictions = []

    def _prepare_img(self, img):
        new_img = img.copy()
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        return new_img
    
    def add_prediction(self, pred_json):
        score = pred_json['score']
        category_id = pred_json['category_id']
        
        # ELSE draw the box
        bbox = pred_json['bbox']
        x1, y1, w, h = list(map(lambda x: int(x), bbox))
        x2 = x1 + w
        y2 = y1 + h

        res_json = {'coords': ((x1, y1), (x2, y2)), 
            'score': score, 
            'category_id': category_id}

        if (score < self.min_score) or (category_id not in self.desired_classes):
            self.undes_predictions.append(res_json)
        else:
            self.des_predictions.append(res_json)

    def annotated(self):
        if len(self) == 0:
            return self.image
        if self.annotated_img is None:
            self.annotated_img = self.image.copy()
            if self.desired > 0:
                for pred in self.des_predictions:
                    self.annotated_img = draw_bbox(self.annotated_img, pred['coords'], 
                        pred['score'], pred['category_id'])
            if self.desired < 2:
                for pred in self.undes_predictions:
                    self.annotated_img = draw_bbox(self.annotated_img, pred['coords'], 
                        pred['score'], pred['category_id'])
        return self.annotated_img
    
    def __len__(self):
        if self.desired == 0:
            return len(self.undes_predictions)
        if self.desired == 1:
            return len(self.des_predictions) + len(self.undes_predictions)
        if self.desired == 2:
            return len(self.des_predictions)
