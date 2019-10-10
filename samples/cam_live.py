import tkinter
import cv2
import PIL.Image, PIL.ImageTk

import cv2
import keyboard


# Create window
window = tkinter.Tk()
window.title("Camera live view")

# initialize the camera
cam = cv2.VideoCapture(0)   # 0 -> index of camera
# Take picture and release camera
s, cv_img = cam.read()
#cam.release()
height, width, no_channels = cv_img.shape

# Create a canvas that can fit the above image
canvas = tkinter.Canvas(window, width = width, height = height)
canvas.pack()
print("Press x to exit")

while not keyboard.is_pressed('x'):
	# Take picture 
	s, cv_img = cam.read()
	cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

	
	if s:    # frame captured without any errors
		# Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
		photo = PIL.ImageTk.PhotoImage(master=canvas,image = PIL.Image.fromarray(cv_img))
		# Add a PhotoImage to the Canvas
		canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)

	window.update_idletasks()
	window.update()

cam.release()
window.quit()
