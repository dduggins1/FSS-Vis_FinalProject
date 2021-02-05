### Import Packages ###

import numpy as np
import pandas as pd
from scipy.stats import describe
from scipy.ndimage.measurements import center_of_mass

from skimage import *
from skimage import color
from skimage import filters
from skimage import measure
from skimage.morphology import binary_closing, binary_dilation, binary_erosion
import cv2

from tkinter import *
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter.messagebox import askquestion

from tqdm.notebook import tqdm
import os
from IPython.display import clear_output
import subprocess
import time
import yaml

from bokeh.plotting import *
from bokeh.models import ColumnDataSource, CustomJS, Slider, Button, DataTable, TableColumn 
from bokeh.models import HoverTool, LinearColorMapper, NumberFormatter, RadioButtonGroup
from bokeh.models import Select, Spinner, RangeSlider, FileInput, MultiLine, PreText
from bokeh.models import Tabs, Panel, MultiSelect, RadioGroup, Div, LabelSet, Label
from bokeh import events
from bokeh.layouts import row, column, layout
from bokeh.io import curdoc, show
from bokeh.themes import Theme
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import EMPTY_LAYOUT, MISSING_RENDERERS
from bokeh.palettes import Category20

# Ignore Warnings about Empty Plots on Initialization
silence(EMPTY_LAYOUT, True)
silence(MISSING_RENDERERS, True)
# Set the Output for the Bokeh Application
output_notebook()

### Define the Bokeh Application ###
# all code is inside this definition except for the call to show at the end #

def bkapp(doc):
    
### Functions ###
    root = Tk()
    root.withdraw()
    # functions for user dialogs

    def open_file(ftype):
        file = askopenfilename(filetypes=ftype,
                               title='Open File',
                               initialdir=os.getcwd())
        #root.destroy()
        return file
    
    def choose_directory():
        #root = Tk()
        #root.withdraw()
        out_dir = askdirectory()
        #root.destroy()
        return out_dir

    def write_output_directory(output_dir):
        #root = Tk()
        #root.withdraw()
        makeDir = askquestion('Make Directory','Output directory not set. Make directory: '
            +output_dir+'? If not, you\'ll be prompted to change directories.',icon = 'warning')
        #root.destroy()
        return makeDir

    def overwrite_file():
        #root = Tk()
        #root.withdraw()
        overwrite = askquestion('Overwrite File','File already exits. Do you want to overwrite?',icon = 'warning')
        #root.destroy()
        return overwrite

    def update_filename():
        filetype = [("Video files", "*.mp4")]
        fname = open_file(filetype)
        if fname:
            #print('Successfully loaded file: '+fname)
            load_data(filename=fname)         

    def change_directory():
        out_dir = choose_directory()
        if out_dir:
            source.data["output_dir"] = [out_dir]
            outDir.text = out_dir
        return out_dir

    # load data from file

    def load_data(filename):
        vidcap = cv2.VideoCapture(filename)
        success,frame = vidcap.read()
        img_tmp,_,__ = cv2.split(frame)
        h,w = np.shape(img_tmp)
        img = np.flipud(img_tmp)
        radio_button_gp.active = 0
        fname = os.path.split(filename)[1]
        input_dir = os.path.split(filename)[0]
        if source.data['output_dir'][0]=='':
            output_dir = os.path.join(input_dir,fname.split('.')[0])
        else:
            output_dir = source.data['output_dir'][0]
        if not os.path.isdir(output_dir):
            makeDir = write_output_directory(output_dir)
            if makeDir=='yes':
                os.mkdir(output_dir)
            else:
                output_dir = change_directory()
        if output_dir:
            source.data = dict(image_orig=[img], image=[img], bin_img=[0],
                x=[0], y=[0], dw=[w], dh=[h], num_contours=[0], roi_coords=[0], 
                img_name=[fname], input_dir=[input_dir], output_dir=[output_dir])
            curr_img = p.select_one({'name':'image'})
            if curr_img:
                p.renderers.remove(curr_img)
            p.image(source=source, image='image', x='x', y='y', dw='dw', dh='dh', color_mapper=cmap,level='image',name='image')      
            p.plot_height=int(h/2)
            p.plot_width=int(w/2)
            #p.add_tools(HoverTool(tooltips=IMG_TOOLTIPS))
            inFile.text = fname
            outDir.text = output_dir
        else:
            print('Cancelled. To continue please set output directory.{:<100}'.format(' '),end="\r")

    # resetting sources for new data or new filters/contours

    def remove_plot():
        source.data["num_contours"]=[0]
        contours_found.text = 'Droplets Detected: 0'
        source_contours.data = dict(xs=[], ys=[])
        source_label.data = dict(x=[], y=[], label=[])

    # apply threshold filter and display binary image

    def apply_filter():
        if source.data['input_dir'][0] == '':
            print('No image loaded! Load image first.{:<100}'.format(' '),end="\r")
        else:
            img = np.squeeze(source.data['image_orig'])
            # remove contours if present
            if source.data["num_contours"]!=[0]:
                remove_plot()
            if radio_button_gp.active == 1:
                thresh = filters.threshold_otsu(img)
                binary = img > thresh
                bin_img = binary*255
                source.data["bin_img"] = [bin_img]
            elif radio_button_gp.active == 2:
                thresh = filters.threshold_isodata(img)
                binary = img > thresh
                bin_img = binary*255
                source.data["bin_img"] = [bin_img]
            elif radio_button_gp.active == 3:
                thresh = filters.threshold_mean(img)
                binary = img > thresh
                bin_img = binary*255
                source.data["bin_img"] = [bin_img]
            elif radio_button_gp.active == 4:
                thresh = filters.threshold_li(img)
                binary = img > thresh
                bin_img = binary*255
                source.data["bin_img"] = [bin_img]
            elif radio_button_gp.active == 5:
                thresh = filters.threshold_yen(img)
                binary = img > thresh
                bin_img = binary*255
                source.data["bin_img"] = [bin_img]
            elif radio_button_gp.active == 6:
                off = offset_spinner.value
                block_size = block_spinner.value
                thresh = filters.threshold_local(img,block_size,offset=off)
                binary = img > thresh
                bin_img = binary*255
                source.data["bin_img"] = [bin_img]
            else:
                bin_img = img
            source.data['image'] = [bin_img]

    # image functions for adjusting the binary image
    
    def close_img():
        if source.data["num_contours"]!=[0]:
            remove_plot()
        if radio_button_gp.active == 0:
            print("Please Select Filter for Threshold{:<100}".format(' '),end="\r")
        else:
            source.data["image"] = source.data["bin_img"]
            img = np.squeeze(source.data['bin_img'])
            closed_img = binary_closing(255-img)*255
            source.data['image'] = [255-closed_img]
            source.data['bin_img'] = [255-closed_img]

    def dilate_img():
        if source.data["num_contours"]!=[0]:
            remove_plot()
        if radio_button_gp.active == 0:
            print("Please Select Filter for Threshold{:<100}".format(' '),end="\r")
        else:
            img = np.squeeze(source.data['bin_img'])
            dilated_img = binary_dilation(255-img)*255
            source.data['image'] = [255-dilated_img]
            source.data['bin_img'] = [255-dilated_img]

    def erode_img():
        if source.data["num_contours"]!=[0]:
            remove_plot()
        if radio_button_gp.active == 0:
            print("Please Select Filter for Threshold{:<100}".format(' '),end="\r")
        else:
            img = np.squeeze(source.data['bin_img'])
            eroded_img = binary_erosion(255-img)*255
            source.data['image'] = [255-eroded_img] 
            source.data['bin_img'] = [255-eroded_img]  

    # the function for identifying closed contours in the image

    def find_contours(level=0.8):
        min_drop_size = contour_rng_slider.value[0]
        max_drop_size = contour_rng_slider.value[1]
        min_dim = 20
        max_dim = 200
        if radio_button_gp.active == 0:
            print("Please Select Filter for Threshold{:<100}".format(' '),end="\r")
        elif source.data['input_dir'][0] == '':
            print('No image loaded! Load image first.{:<100}'.format(' '),end="\r")
        else:
            img = np.squeeze(source.data['bin_img'])
            h,w = np.shape(img)        
            contours = measure.find_contours(img, level)
            length_cnt_x = [cnt[:,1] for cnt in contours if np.shape(cnt)[0] < max_drop_size 
                             and np.shape(cnt)[0] > min_drop_size]
            length_cnt_y = [cnt[:,0] for cnt in contours if np.shape(cnt)[0] < max_drop_size 
                             and np.shape(cnt)[0] > min_drop_size]
            matched_cnt_x = []
            matched_cnt_y = []
            roi_coords = []
            label_text = []
            label_y = np.array([])
            count=0
            for i in range(len(length_cnt_x)):
                cnt_x = length_cnt_x[i]
                cnt_y = length_cnt_y[i]
                width = np.max(cnt_x)-np.min(cnt_x)
                height = np.max(cnt_y)-np.min(cnt_y)
                if width>min_dim and width<max_dim and height>min_dim and height<max_dim:
                    matched_cnt_x.append(cnt_x)
                    matched_cnt_y.append(cnt_y)
                    roi_coords.append([round(width),round(height),round(np.min(cnt_x)),round(h-np.max(cnt_y))])
                    label_text.append(str(int(count)+1))
                    label_y = np.append(label_y,np.max(cnt_y))
                    count+=1
            curr_contours = p.select_one({'name':'overlay'})
            if curr_contours:
                p.renderers.remove(curr_contours)
            #if source.data["num_contours"]==[0]:
                #remove_plot()
                #p.image(source=source, image='image_orig', x=0, y=0, dw=w, dh=h, color_mapper=cmap, name='overlay',level='underlay')      
            source.data["image"] = source.data["image_orig"]
            source.data["num_contours"] = [count]
            #source.data["cnt_x"] = [matched_cnt_x]
            #source.data["cnt_y"] = [matched_cnt_y]
            source.data["roi_coords"] = [roi_coords]
            source_contours.data = dict(xs=matched_cnt_x, ys=matched_cnt_y)
            p.multi_line(xs='xs',ys='ys',source=source_contours, color=(255,127,14),line_width=2, name="contours",level='glyph')
            if len(np.array(roi_coords).shape)<2:
                if len(np.array(roi_coords)) <4:
                    print('No contours found. Try adjusting parameters or filter for thresholding.{:<100}'.format(' '),end="\r")
                    source_label.data = dict(x=[],y=[],label=[])
                else:
                    source_label.data = dict(x=np.array(roi_coords)[2], y=label_y, label=label_text)
            else:
                source_label.data = dict(x=np.array(roi_coords)[:,2], y=label_y, label=label_text)
            contours_found.text = 'Droplets Detected: '+str(len(matched_cnt_x))

    # write the contours and parameters to files

    def export_ROIs():
        if source.data["num_contours"]==[0]:
            print("No Contours Found! Find contours first.{:<100}".format(' '),end="\r")
        else:
            hdr = 'threshold filter,contour min,contour max'
            thresh_filter = radio_button_gp.active
            cnt_min = contour_rng_slider.value[0]
            cnt_max = contour_rng_slider.value[1]
            params = [thresh_filter,cnt_min,cnt_max]
            if radio_button_gp.active == 6:
                off = offset_spinner.value
                block_size = block_spinner.value
                hdr = hdr + ',local offset,local block size'
                params.append(off,block_size)
            params_fname = 'ContourParams.csv'
            params_out = os.path.join(source.data['output_dir'][0],params_fname)
            overwrite = 'no'
            if os.path.exists(params_out):
                overwrite = overwrite_file()
            if overwrite == 'yes' or not os.path.exists(params_out):
                np.savetxt(params_out,np.array([params]),delimiter=',',header=hdr,comments='')
            roi_coords = np.array(source.data["roi_coords"][0])
            out_fname = 'ROI_coords.csv'
            out_fullpath = os.path.join(source.data['output_dir'][0],out_fname)
            if overwrite == 'yes' or not os.path.exists(out_fullpath):
                hdr = 'width,height,x,y'
                np.savetxt(out_fullpath,roi_coords,delimiter=',',header=hdr,comments='')
                print('Successfully saved ROIs coordinates as: '+out_fullpath,end='\r')
                source.data['roi_coords'] = [roi_coords]

    # function for loading previously made files or error handling for going out of order

    def check_ROI_files():
        coords_file = os.path.join(source.data["output_dir"][0],'ROI_coords.csv')
        n_cnt_curr = source.data["num_contours"][0]
        roi_coords_curr = source.data["roi_coords"][0]
        if os.path.exists(coords_file):
            df_tmp=pd.read_csv(coords_file, sep=',')
            roi_coords = np.array(df_tmp.values)
            n_cnt = len(roi_coords)
            if n_cnt_curr != n_cnt or not np.array_equal(roi_coords_curr,roi_coords):
                print('Current ROIs are different from saved ROI file! ROIs from saved file will be used instead and plot updated.',end="\r")
            source.data["num_contours"] = [n_cnt]
            source.data["roi_coords"] = [roi_coords]
            params_file = os.path.join(source.data['output_dir'][0],'ContourParams.csv')
            params_df = pd.read_csv(params_file)
            thresh_ind = params_df["threshold filter"].values[0]
            radio_button_gp.active = int(thresh_ind)
            if thresh_ind == 6:
                offset_spinner.value = int(params_df["local offset"].values[0])
                block_spinner.value = int(params_df["local block size"].values[0])
            contour_rng_slider.value = tuple([int(params_df["contour min"].values[0]),int(params_df["contour max"].values[0])])
            find_contours()
        else:
            print("ROI files not found! Check save directory or export ROIs.{:<100}".format(' '),end="\r")

    # use FFMPEG to crop out regions from original mp4 and save to file

    def create_ROI_movies():
        if source.data['input_dir'][0] == '':
            print('No image loaded! Load image first.{:<100}'.format(' '),end="\r")
        else:
            check_ROI_files()
            side = 100 # for square ROIs, replace first two crop parameters with side & uncomment
            padding = 20
            roi_coords_file = os.path.join(source.data['output_dir'][0],'ROI_coords.csv')
            if source.data["num_contours"]==[0]:
                print("No contours found! Find contours first.{:<100}".format(' '),end="\r")
            elif not os.path.exists(roi_coords_file):
                print("ROI file does not exist! Check save directory or export ROIs.{:<100}".format(' '),end="\r")
            else:
                print('Creating Movies...{:<100}'.format(' '),end="\r")
                pbar = tqdm(total=source.data["num_contours"][0])
                for i in range(source.data["num_contours"][0]):
                    roi_coords = np.array(source.data["roi_coords"][0])
                    inPath = os.path.join(source.data['input_dir'][0],source.data['img_name'][0])
                    #out_fname = source.data['filename'][0].split('.')[0] +'_ROI'+str(i+1)+'.mp4'
                    out_fname = 'ROI'+str(i+1)+'.mp4'
                    outPath = os.path.join(source.data['output_dir'][0],out_fname)
                    #command = f"ffmpeg -i \'{(inPath)}\' -vf \"crop={(roi_coords[i,0]+padding*2)}:{(roi_coords[i,1]+padding*2)}:{(roi_coords[i,2]-padding)}:{(roi_coords[i,3]+padding)}\" -y \'{(outPath)}\'"
                    command = f"ffmpeg -i \'{(inPath)}\' -vf \"crop={side}:{side}:{(roi_coords[i,2]-padding)}:{(roi_coords[i,3]-padding)}\" -y \'{(outPath)}\'"
                    overwrite = 'no'
                    if os.path.exists(outPath):
                        overwrite = overwrite_file()
                    if overwrite == 'yes' or not os.path.exists(outPath):
                        saved = subprocess.check_call(command,shell=True)
                        if saved != 0:
                            print('An error occurred while creating movies! Check terminal window.{:<100}'.format(' '),end="\r")
                    pbar.update()

    # change the display range on images from slider values

    def update_image():
        cmap.low = display_range_slider.value[0]
        cmap.high = display_range_slider.value[1]

    # create statistics files for each mp4 region specific file

    def process_ROIs():
        if source.data['input_dir'][0] == '':
            print('No image loaded! Load image first.{:<100}'.format(' '),end="\r")
        else:
            check_ROI_files()
            hdr = 'roi,time,area,mean,variance,min,max,median,skewness,kurtosis,rawDensity,COMx,COMy'
            cols = hdr.split(',')
            all_stats = np.zeros((0,13))
            n_cnt = source.data["num_contours"][0]
            if n_cnt == 0:
                print("No contours found! Find contours first.{:<100}".format(' '),end="\r")
            for i in range(n_cnt): 
                #in_fname = source.data['filename'][0].split('.')[0] +'_ROI'+str(i+1)+'.mp4'
                in_fname = 'ROI'+str(i+1)+'.mp4'
                inPath = os.path.join(source.data['output_dir'][0],in_fname)
                #out_fname = source.data['filename'][0].split('.')[0] +'_ROI'+str(i+1)+'_stats.csv'
                out_fname = 'stats_ROI'+str(i+1)+'.csv'
                outPath = os.path.join(source.data['output_dir'][0],out_fname)
                if not os.path.exists(inPath):
                    print('ROI movie not found! Create ROI movie first.{:<100}'.format(' '),end="\r")
                    break
                vidcap = cv2.VideoCapture(inPath)
                last_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
                if i==0:
                    pbar = tqdm(total=last_frame*n_cnt)
                success,frame = vidcap.read()
                img_tmp,_,__ = cv2.split(frame)
                h,w = np.shape(img_tmp)
                img = np.zeros((last_frame,h,w))
                img_stats = np.zeros((last_frame,13))
                stats = describe(img_tmp,axis=None)
                median = np.median(img_tmp)
                density = np.sum(img_tmp)
                cx, cy = center_of_mass(img_tmp)
                img_stats[0,0:13] = [i,0,stats.nobs,stats.mean,stats.variance,
                        stats.minmax[0],stats.minmax[1],median,stats.skewness,
                        stats.kurtosis,density,cx,cy]
                img[0,:,:] = np.flipud(img_tmp)
                pbar.update()
                overwrite = 'no'
                if os.path.exists(outPath):
                    overwrite = overwrite_file()
                    if overwrite=='no':
                        pbar.update(last_frame-1)
                if overwrite == 'yes' or not os.path.exists(outPath):
                    for j in range(1,last_frame):
                        vidcap.set(1, j)
                        success,frame = vidcap.read()
                        img_tmp,_,__ = cv2.split(frame)
                        stats = describe(img_tmp,axis=None)
                        t = j*5/60
                        density = np.sum(img_tmp)
                        cx, cy = center_of_mass(img_tmp)
                        median = np.median(img_tmp)
                        img_stats[j,0:13] = [i,t,stats.nobs,stats.mean,stats.variance,
                                stats.minmax[0],stats.minmax[1],median,stats.skewness,
                                stats.kurtosis,density,cx,cy]
                        img[j,:,:] = np.flipud(img_tmp)
                        pbar.update(1)
                    all_stats = np.append(all_stats,img_stats,axis=0)
                    np.savetxt(outPath,img_stats,delimiter=',',header=hdr,comments='')
                if i==(n_cnt-1):
                    df = pd.DataFrame(all_stats,columns=cols)
                    group = df.groupby('roi')
                    for i in range(len(group)):
                        sources_stats[i] = ColumnDataSource(group.get_group(i))

    # load statistics CSVs and first ROI mp4 files and display in plots

    def load_ROI_files():
        if source.data['input_dir'][0] == '':
            print('No image loaded! Load image first.{:<100}'.format(' '),end="\r")
        else:
            check_ROI_files()
            n_cnt = source.data["num_contours"][0]
            basepath = os.path.join(source.data["output_dir"][0],'stats')
            all_files = [basepath+'_ROI'+str(i+1)+'.csv' for i in range(n_cnt)]
            files_exist = [os.path.exists(f) for f in all_files]
            if all(files_exist) and n_cnt != 0:
                df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
                group = df.groupby('roi')
                OPTIONS = []
                LABELS = []
                pbar = tqdm(total=len(stats)*len(group))
                j=0
                colors_ordered = list(Category20[20])
                idx_reorder = np.append(np.linspace(0,18,10),np.linspace(1,19,10))
                idx = idx_reorder.astype(int)
                colors = [colors_ordered[i] for i in idx]
                for roi, df_roi in group:
                    sources_stats[roi] = ColumnDataSource(df_roi)
                    OPTIONS.append([str(int(roi)+1),'ROI '+(str(int(roi)+1))])
                    LABELS.append('ROI '+str(int(roi)+1))
                    color = colors[j]
                    j+=1
                    if j>=20:
                        j=0
                    for i in range(3,len(df.columns)):
                        name = 'ROI '+str(int(roi)+1)
                        plot_check = p_stats[i-3].select_one({'name':name})
                        if not plot_check:
                            p_stats[i-3].line(x='time',y=str(df.columns[i]),source=sources_stats[roi],
                                name=name,visible=False,line_color=color)
                            p_stats[i-3].xaxis.axis_label = "Time [h]"
                            p_stats[i-3].yaxis.axis_label = str(df.columns[i])
                            p_stats[i-3].add_tools(HoverTool(tooltips=TOOLTIPS))
                            p_stats[i-3].toolbar_location = "left"
                        pbar.update(1)
                ROI_multi_select.options = OPTIONS 
                ROI_multi_select.value = ["1"]
                ROI_movie_radio_group.labels = LABELS
                ROI_movie_radio_group.active = 0
            else:
                print('Not enough files! Check save directory or calculate new stats.{:<100}'.format(' '),end="\r")

    # show/hide curves from selected/deselected labels for ROIs in statistics plots

    def update_ROI_plots():
        n_cnt = source.data["num_contours"][0]
        pbar = tqdm(total=len(stats)*n_cnt)
        for j in range(n_cnt):
            for i in range(len(stats)):
                name = 'ROI '+str(int(j)+1)
                glyph = p_stats[i].select_one({'name': name})
                if str(j+1) in ROI_multi_select.value:
                    glyph.visible = True
                else:
                    glyph.visible = False
                pbar.update(1)

    # load and display the selected ROI's mp4

    def load_ROI_movie():
        idx = ROI_movie_radio_group.active
        in_fname = 'ROI'+str(idx+1)+'.mp4'
        inPath = os.path.join(source.data['output_dir'][0],in_fname)
        if not os.path.exists(inPath):
            print('ROI movie not found! Check save directory or create ROI movie.',end="\r")
        else:
            old_plot = p_ROI.select_one({'name': sourceROI.data['img_name'][0]})
            if old_plot:
                p_ROI.renderers.remove(old_plot)
            vidcap = cv2.VideoCapture(inPath)
            last_frame = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            ROI_movie_slider.end = (last_frame-1)*5/60
            ROI_movie_slider.value = 0
            vidcap.set(1, 0)
            success,frame = vidcap.read()
            img_tmp,_,__ = cv2.split(frame)
            h,w = np.shape(img_tmp)
            img = np.flipud(img_tmp)
            name = 'ROI'+str(idx+1)
            sourceROI.data = dict(image=[img],x=[0], y=[0], dw=[w], dh=[h],
                img_name=[name])
            p_ROI.image(source=sourceROI, image='image', x='x', y='y', 
               dw='dw', dh='dh', color_mapper=cmap, name='img_name')

    # change the displayed frame from slider movement

    def update_ROI_movie():
        frame_idx = round(ROI_movie_slider.value*60/5)
        in_fname = sourceROI.data['img_name'][0]+'.mp4'
        inPath = os.path.join(source.data['output_dir'][0],in_fname)
        vidcap = cv2.VideoCapture(inPath)
        vidcap.set(1, frame_idx)
        success,frame = vidcap.read()
        img_tmp,_,__ = cv2.split(frame)
        img = np.flipud(img_tmp)
        sourceROI.data['image'] = [img]

    # the following 2 functions are used to animate the mp4

    def update_ROI_slider():
        time = ROI_movie_slider.value + 5/60
        end = ROI_movie_slider.end
        if time > end:
            animate_ROI_movie()
        else:
            ROI_movie_slider.value = time
        return callback_id

    def animate_ROI_movie():
        global callback_id
        if ROI_movie_play_button.label == '► Play':
            ROI_movie_play_button.label = '❚❚ Pause'
            callback_id = curdoc().add_periodic_callback(update_ROI_slider, 10)
        else:
            ROI_movie_play_button.label = '► Play'
            curdoc().remove_periodic_callback(callback_id)
        return callback_id

### Application Content ###

    # main plot for segmentation and contour finding

    cmap = LinearColorMapper(palette="Greys256", low=0, high=255)
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"
    IMG_TOOLTIPS = [('name', "@img_name"),("x", "$x"),("y", "$y"),("value", "@image")]

    source = ColumnDataSource(data=dict(image=[0],bin_img=[0],image_orig=[0],
            x=[0], y=[0], dw=[0], dh=[0], num_contours=[0], roi_coords=[0],
            input_dir=[''],output_dir=[''],img_name=['']))
    source_label = ColumnDataSource(data=dict(x=[0], y=[0], label=['']))
    source_contours = ColumnDataSource(data=dict(xs=[0], ys=[0]))

    roi_labels = LabelSet(x='x', y='y', text='label',source=source_label, 
        level='annotation',text_color='white',text_font_size='12pt')

    # create a new plot and add a renderer
    p = figure(tools=TOOLS, toolbar_location=("right"))
    p.add_layout(roi_labels)
    p.x_range.range_padding = p.y_range.range_padding = 0

    # turn off gridlines
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.visible = False


    # ROI plots 

    sourceROI = ColumnDataSource(data=dict(image=[0],
            x=[0], y=[0], dw=[0], dh=[0], img_name=[0]))
    sources_stats = {}

    TOOLTIPS = [('name','$name'),('time', '@time'),('stat', "$y")]
    stats = np.array(['mean','var','min','max','median','skew','kurt','rawDensity','COMx','COMy'])
    p_stats = []
    tabs = []
    for i in range(len(stats)):
        p_stats.append(figure(tools=TOOLS, plot_height=300, plot_width=600))
        p_stats[i].x_range.range_padding = p_stats[i].y_range.range_padding = 0
        tabs.append(Panel(child=p_stats[i], title=stats[i]))

    # create a new plot and add a renderer
    p_ROI = figure(tools=TOOLS, toolbar_location=("right"), plot_height=300, plot_width=300)
    p_ROI.x_range.range_padding = p_ROI.y_range.range_padding = 0  

    # turn off gridlines
    p_ROI.xgrid.grid_line_color = p_ROI.ygrid.grid_line_color = None
    p_ROI.axis.visible = False


    # Widgets - Buttons, Sliders, Text, Etc.

    intro = Div(text="""<h2>Droplet Recognition and Analysis with Bokeh</h2> 
        This application is designed to help segment a grayscale image into 
        regions of interest (ROIs) and perform analysis on those regions.<br>
        <h4>How to Use This Application:</h4>
        <ol>
        <li>Load in a grayscale mp4 file and choose a save directory.</li>
        <li>Apply various filters for thresholding. Use <b>Close</b>, <b>Dilate</b> 
        and <b>Erode</b> buttons to adjust each binary image further.</li>
        <li>Use <b>Find Contours</b> button to search the image for closed shape. 
        The <b>Contour Size Range</b> slider will change size of the perimeter to
        be identified. You can apply new thresholds and repeat until satisfied with
        the region selection. Total regions detected is displayed next to
        the button.</li>
        <li>When satisfied, use <b>Export ROIs</b> to write ROI locations and 
        contour finding parameters to file.</li>
        <li><b>Create ROI Movies</b> to write mp4s of the selected regions.</li>
        <li>Use <b>Calculate ROI Stats</b> to perform calculations on the 
        newly created mp4 files.</li>
        <li>Finally, use <b>Load ROI Files</b> to load in the data that you just
        created and view the plots. The statistics plots can be overlaid by 
        selecting multiple labels. Individual ROI mp4s can be animated or you can
        use the slider to move through the frames.</li>
        </ol>
        Note: messages and progress bars are displayed below the GUI.""",
        style={'font-size':'10pt'},width=1000)

    file_button = Button(label="Choose File",button_type="primary")
    file_button.on_click(update_filename)
    inFile = PreText(text='Input File:\n'+source.data["img_name"][0], background=(255,255,255,0.5), width=500)

    filter_LABELS = ["Original","OTSU", "Isodata", "Mean", "Li","Yen","Local"]
    radio_button_gp = RadioButtonGroup(labels=filter_LABELS, active=0, width=600)
    radio_button_gp.on_change('active', lambda attr, old, new: apply_filter())
    
    offset_spinner = Spinner(low=0, high=500, value=1, step=1, width=100, title="Local: Offset",
                            background=(255,255,255,0.5))
    offset_spinner.on_change('value', lambda attr, old, new: apply_filter())
    block_spinner = Spinner(low=1, high=101, value=25, step=2, width=100, title="Local: Block Size",
                           background=(255,255,255,0.5))
    block_spinner.on_change('value', lambda attr, old, new: apply_filter())
    
    closing_button = Button(label="Close",button_type="default", width=100)
    closing_button.on_click(close_img)
    dilation_button = Button(label="Dilate",button_type="default", width=100)
    dilation_button.on_click(dilate_img)
    erosion_button = Button(label="Erode",button_type="default", width=100)
    erosion_button.on_click(erode_img)

    contour_rng_slider = RangeSlider(start=10, end=500, value=(200,350), step=1, width=300, 
            title="Contour Size Range", background=(255,255,255,0.5), bar_color='gray')
    contour_button = Button(label="Find Contours", button_type="success")
    contour_button.on_click(find_contours)
    contours_found = PreText(text='Droplets Detected: '+str(source.data["num_contours"][0]), background=(255,255,255,0.5))
    
    exportROIs_button = Button(label="Export ROIs", button_type="success", width=200)
    exportROIs_button.on_click(export_ROIs)    

    changeDir_button = Button(label="Change Directory",button_type="primary", width=150)
    changeDir_button.on_click(change_directory)
    outDir = PreText(text='Save Directory:\n'+source.data["output_dir"][0], background=(255,255,255,0.5), width=500)

    create_ROIs_button = Button(label="Create ROI Movies",button_type="success", width=200)
    create_ROIs_button.on_click(create_ROI_movies)

    process_ROIs_button = Button(label="Calculate ROI Stats",button_type="success")
    process_ROIs_button.on_click(process_ROIs)

    display_rng_text = figure(title="Display Range", title_location="left", 
                        width=40, height=300, toolbar_location=None, min_border=0, 
                        outline_line_color=None)
    display_rng_text.title.align="center"
    display_rng_text.title.text_font_size = '10pt'
    display_rng_text.x_range.range_padding = display_rng_text.y_range.range_padding = 0

    display_range_slider = RangeSlider(start=0, end=255, value=(0,255), step=1, 
        orientation='vertical', direction='rtl', 
        bar_color='gray', width=40, height=300, tooltips=True)
    display_range_slider.on_change('value', lambda attr, old, new: update_image())

    load_ROIfiles_button = Button(label="Load ROI Files",button_type="primary")
    load_ROIfiles_button.on_click(load_ROI_files)

    ROI_multi_select = MultiSelect(value=[], width=100, height=300)
    ROI_multi_select.on_change('value', lambda attr, old, new: update_ROI_plots())

    ROI_movie_radio_group = RadioGroup(labels=[],width=60)
    ROI_movie_radio_group.on_change('active', lambda attr, old, new: load_ROI_movie())
    ROI_movie_slider = Slider(start=0,end=100,value=0,step=5/60,title="Time [h]", width=280)
    ROI_movie_slider.on_change('value', lambda attr, old, new: update_ROI_movie())

    callback_id = None

    ROI_movie_play_button = Button(label='► Play',width=50)
    ROI_movie_play_button.on_click(animate_ROI_movie)

# initialize some data without having to choose file
    # fname = os.path.join(os.getcwd(),'data','Droplets.mp4')
    # load_data(filename=fname)


### Layout & Initialize application ###

    ROI_layout = layout([
        [ROI_movie_radio_group, p_ROI],
        [ROI_movie_slider,ROI_movie_play_button]
        ])

    app = layout(children=[
        [intro],
        [file_button,inFile],
        [radio_button_gp, offset_spinner, block_spinner],
        [closing_button, dilation_button, erosion_button],
        [contour_rng_slider, contour_button, contours_found],
        [exportROIs_button, outDir, changeDir_button],
        [create_ROIs_button, process_ROIs_button],
        [display_rng_text, display_range_slider, p],
        [load_ROIfiles_button],
        [ROI_layout, ROI_multi_select, Tabs(tabs=tabs)]
    ])
    
    doc.add_root(app)

show(bkapp)