
	Data breaches occur all over the internet while transferring messages/information from one end to another end. By constantly enhancing the security architecture we can improve the factors such as Confidentiality, Integrity, Accountability, Availability while transferring the data from sender to receiver. This can be achieved by using the concept of Steganography. Steganography is a scheme to conceal a secret file inside another means of information, also known as a cover, which can be retrieved at the destination. (Steganography is hiding the information in a multimedia content such as Image, Audio, Video etc.). Here we have taken Image Steganography, the information is concealed in an Image without detection of a hidden message. Here we implemented the concept of Least Bit Significant method. The LSB value matches with the binary value of the input message and embed the value within the pixels of the image by making sure that the values of pixels are not affect the image whole content. By converting the image to array values of RGB and the message is embedded in the pixel value.  To make sure that there is a message in the sent image, we make sure that there is pattern(header) enclosed in the image. This concept is known as Framing. Framing is concept where data are transferred as a stream of bits from one point to another. By detecting the presence of header, the receiver can decrypt the input. For additional layer of Security, we use the International Data Encryption Algorithm (IDEA) concealing the information.