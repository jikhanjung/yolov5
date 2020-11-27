from pathlib import Path

fieldnames = [ 'folder_name', 'image_name', 'image_width', 'image_height', 'class_id', 'fin_index', 'center_x', 'center_y', 'width', 'height', 'confidence', 
               'is_fin', 'image_datetime', 'location', 'latitude', 'longitude','map_datum','dolfin_id', 'observed_by', 'created_by', 'created_on', 'modified_by', 'modified_on', 'comment' ]

class DolfinRecord:
    def __init__(self,info={}):
        self.folder_name = ''
        self.image_name = ''
        self.image_width = 0
        self.image_height = 0
        self.class_id = 0
        self.fin_index = -1
        self.center_x = -1
        self.center_y = -1
        self.width = -1
        self.height = -1
        self.confidence = 0
        self.is_fin = True
        self.image_datetime = ''
        #self.image_time = ''
        self.location = ''
        self.latitude = ''
        self.longitude = ''
        self.map_datum = ''
        self.dolfin_id = ''
        self.observed_by = ''
        self.created_by = ''
        self.created_on = ''
        self.modified_by = ''
        self.modified_on = ''
        self.comment = ''

        if( len( info ) > 0 ):
            self.set_info( info )
        #print( "info:", info['modified_on'] )
    
    #def fieldnames(self):
    def get_detection_info(self):
        return self.class_id, self.center_x, self.center_y, self.width, self.height, self.confidence

    def get_info(self):
        info_hash = { 
            'folder_name': self.folder_name,
            'image_name': self.image_name,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'class_id': self.class_id,
            'fin_index': self.fin_index,
            'center_x': self.center_x,
            'center_y': self.center_y,
            'width': self.width,
            'height': self.height,
            'confidence': self.confidence,
            'is_fin': self.is_fin,
            'image_datetime': self.image_datetime,
            'location': self.location,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'map_datum': self.map_datum,
            'dolfin_id': self.dolfin_id,
            'observed_by': self.observed_by,
            'created_by': self.created_by,
            'created_on': self.created_on,
            'modified_by': self.modified_by,
            'modified_on': self.modified_on,
            'comment': self.comment,
        }
        return info_hash

    def set_imagesize(self, image_width, image_height ):
        self.image_width = image_width
        self.image_height = image_height

    def get_x1y1x2y2( self ):
        result = self.get_x1y1x2y2_normalized()
        if int(self.image_width) > 0 and int(self.image_height) > 0:
            result['x1'] *= int(self.image_width)
            result['x2'] *= int(self.image_width)
            result['y1'] *= int(self.image_height)
            result['y2'] *= int(self.image_height)
            result['width'] = result['x2'] - result['x1']
            result['height'] = result['y2'] - result['y1']
        return result

    def get_x1y1x2y2_normalized( self ):
        return { 'x1': self.center_x - 0.5 * self.width,  'x2': self.center_x + 0.5 * self.width, 
                 'y1': self.center_y - 0.5 * self.height, 'y2': self.center_y + 0.5 * self.height }

    def get_area( self ):
        coord = self.get_x1y1x2y2()
        return ( coord['x2'] - coord['x1'] ) * ( coord['y2'] - coord['y1'] )
    
    def get_intersection( self, rec ):
        coord1 = self.get_x1y1x2y2()
        coord2 = rec.get_x1y1x2y2()
        intersection = {}
        if coord1['x1'] > coord2['x2'] or coord1['x2'] < coord2['x1']:
            return 0
        if coord1['y1'] > coord2['y2'] or coord1['y2'] < coord2['y1']:
            return 0
        intersection['x1'] = max( coord1['x1'], coord2['x1'] )
        intersection['x2'] = min( coord1['x2'], coord2['x2'] )
        intersection['y1'] = max( coord1['y1'], coord2['y1'] )
        intersection['y2'] = min( coord1['y2'], coord2['y2'] )
        return ( intersection['x2'] - intersection['x1'] ) * ( intersection['y2'] - intersection['y1'] )

    def get_iou( self, rec ):
        if self.confidence < 0 or rec.confidence < 0:
            return 0
        intersection_area = self.get_intersection( rec )
        union_area = self.get_area() + rec.get_area() - intersection_area
        iou = intersection_area / union_area
        return iou        

    def get_itemname(self):
        if self.confidence < 0:
            return self.image_name
        else:
            return self.image_name + '-' + str( self.fin_index ) 

    def get_itemname_with_dolfin_id(self):
        if self.confidence < 0:
            return self.get_itemname()
        else:
            name = self.get_itemname()
            if( self.is_fin == False ):
                name += ' (No Fin)'
            elif( self.dolfin_id != '' ):
                name += ' (' + self.dolfin_id + ')'
            return name

    def get_finname(self):
        return self.get_iconfile_stem()
    
    def get_iconfile_stem(self):
        imagename_stem = str(Path(self.image_name).stem)

        if self.confidence < 0:
            return imagename_stem
        else:
            return '{}-{:02d}'.format( imagename_stem, int(self.fin_index) )
    def get_decimal_latitude_longitude(self):
        lat, lon = -1, -1
        if self.latitude == '' or self.longitude == '':
            return lat, lon

        lat_str, lon_str = self.latitude, self.longitude

        deg, rest = lat_str.split("°")
        minute, ns = rest.split("'")
        plus_minus = 1
        if ns == 'S':
            plus_minus = -1
        lat = (int(deg) + float(minute) / 60) * plus_minus

        deg, rest = lon_str.split("°")
        minute, EW = rest.split("'")
        lon = (int(deg) + float(minute) / 60) * plus_minus
        return lat, lon

    def find_matching_record( self, record_list ):
        dist_list = []
        coords_list = []
        max_iou = 0
        matching_rec = None
        for rec in record_list:
            iou = self.get_iou( rec )
            if max_iou < iou:
                matching_rec = rec
                max_iou = iou
        #if matching_rec != None:
        return matching_rec, max_iou

    def set_info( self, info ):

        if( 'folder_name' in info.keys() ):
            self.folder_name = info['folder_name']
        if( 'image_name' in info.keys() ):
            self.image_name = info['image_name']
        if( 'image_width' in info.keys() ):
            self.image_width = int(info['image_width'])
        if( 'image_height' in info.keys() ):
            self.image_height = int(info['image_height'])
        if( 'class_id' in info.keys() ):
            self.class_id = int(info['class_id'])
        if( 'fin_index' in info.keys() ):
            self.fin_index = int(info['fin_index'])
        if( 'center_x' in info.keys() ):
            self.center_x = float(info['center_x'])
        if( 'center_y' in info.keys() ):
            self.center_y = float(info['center_y'])
        if( 'width' in info.keys() ):
            self.width = float(info['width'])
        if( 'height' in info.keys() ):
            self.height = float(info['height'])
        if( 'confidence' in info.keys() ):
            self.confidence = float(info['confidence'])
        if( 'is_fin' in info.keys() ):
            if( str(info['is_fin']).lower() == 'true' or info['is_fin'] == True):
                self.is_fin = True
            else:
                self.is_fin = False
            #print( "record is_fin:", info['image_name'], info['fin_index'], info['is_fin'], self.is_fin)
        if( 'image_datetime' in info.keys() ):
            self.image_datetime = info['image_datetime']
        #if( 'image_time' in info.keys() ):
        #    self.image_time = info['image_time']
        if( 'location' in info.keys() ):
            self.location = info['location']
        if( 'latitude' in info.keys() ):
            self.latitude = info['latitude']
        if( 'longitude' in info.keys() ):
            self.longitude = info['longitude']
        if( 'map_datum' in info.keys() ):
            self.map_datum = info['map_datum']
        if( 'dolfin_id' in info.keys() ):
            self.dolfin_id = info['dolfin_id']
        if( 'observed_by' in info.keys() ):
            self.observed_by = info['observed_by']
        if( 'created_by' in info.keys() ):
            self.created_by = info['created_by']
        if( 'created_on' in info.keys() ):
            self.created_on = info['created_on']
        if( 'modified_by' in info.keys() ):
            self.modified_by = info['modified_by']
        if( 'modified_on' in info.keys() ):
            self.modified_on = info['modified_on']
            #print('modified_on setting:', info['modified_on'], self.modified_on)
        if( 'comment' in info.keys() ):
            self.comment = info['comment']
