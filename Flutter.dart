import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:camera/camera.dart';
import 'package:path/path.dart' show join;
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Emotion Detection',
      theme: ThemeData(primarySwatch: Colors.green),
      home: ImageProcessingScreen(),
    );
  }
}

class ImageProcessingScreen extends StatefulWidget {
  @override
  _ImageProcessingScreenState createState() => _ImageProcessingScreenState();
}

class _ImageProcessingScreenState extends State<ImageProcessingScreen> {
  File? _imageFile;
  CameraController? _cameraController;
  final picker = ImagePicker();
  String responseData = "";
  bool isLoading = false;
  bool isLiveCamera = false;
  late Timer _timer;
  List<CameraDescription>? cameras;
  int selectedCameraIdx = 0;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    cameras = await availableCameras();
    if (cameras != null && cameras!.isNotEmpty) {
      _cameraController = CameraController(cameras![selectedCameraIdx], ResolutionPreset.medium);
      await _cameraController!.initialize();
      setState(() {});
    }
  }

  Future<void> _toggleCamera() async {
    if (cameras != null && cameras!.length > 1) {
      setState(() {
        selectedCameraIdx = (selectedCameraIdx + 1) % cameras!.length;
      });

      await _cameraController!.dispose();
      _cameraController = CameraController(cameras![selectedCameraIdx], ResolutionPreset.medium);
      await _cameraController!.initialize();
      setState(() {});
    }
  }

  Future<void> getImageEmotion() async {
    final pickedFile = await picker.getImage(source: ImageSource.gallery);

    setState(() {
      if (pickedFile != null) {
        _imageFile = File(pickedFile.path);
        sendImageToServer(_imageFile!);
        isLoading = true;
      } else {
        print('No image selected.');
      }
    });
  }

  Future<void> predictLiveCamera() async {
    setState(() {
      isLiveCamera = !isLiveCamera;
    });

    if (isLiveCamera) {
      await _initializeCamera();
      _timer = Timer.periodic(Duration(milliseconds: 500), (timer) {
        if (_cameraController!.value.isInitialized) {
          processCameraImage();
        }
      });
    } else {
      _timer.cancel();
      _cameraController?.dispose();
      _imageFile = null;
    }
  }

  Future<void> processCameraImage() async {
    try {
      final Directory extDir = await getTemporaryDirectory();
      final String dirPath = '${extDir.path}/Pictures/';
      await Directory(dirPath).create(recursive: true);
      final String filePath = '$dirPath${DateTime.now()}.png';

      if (_cameraController != null) {
        XFile imageFile = await _cameraController!.takePicture();

        File image = File(imageFile.path);
        sendImageToServer(image);
      }
    } catch (e) {
      print("Error capturing image: $e");
    }
  }

  Future<void> sendImageToServer(File imageFile) async {
    var request = http.MultipartRequest('POST', Uri.parse('your localhost address'),);
    request.files.add(await http.MultipartFile.fromPath('image', imageFile.path));

    try {
      var response = await request.send();
      if (response.statusCode == 200) {
        responseData = await response.stream.bytesToString();
        print('Response from server: $responseData');
        setState(() {
          responseData = responseData.isNotEmpty ? responseData : "Unknown";
        });
      } else {
        print('Failed to send image. Status code: ${response.statusCode}');
      }
    } catch (e) {
      print('Error sending image: $e');
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  @override
  void dispose() {
    _timer.cancel();
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Emotion Detection'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            if (isLiveCamera || _imageFile != null)
              isLoading
                  ? CircularProgressIndicator()
                  : Text(
                'Emotion: ${responseData.isEmpty ? "Awaiting response" : responseData}',
                style: TextStyle(fontSize: 20),
              ),
            if (!isLiveCamera)
              _imageFile != null
                  ? Image.file(
                _imageFile!,
                height: 350,
              )
                  : Text('No image selected'),
            if (isLiveCamera)
              _cameraController != null && _cameraController!.value.isInitialized
                  ? Stack(
                children: [
                  Container(
                    width: 340,
                    height: 440,
                    child: ClipRect(
                      child: OverflowBox(
                        alignment: Alignment.center,
                        child: FittedBox(
                          fit: BoxFit.cover,
                          child: SizedBox(
                            width: _cameraController!.value.previewSize!.height,
                            height: _cameraController!.value.previewSize!.width,
                            child: Transform.scale(
                              scale: _cameraController!.value.aspectRatio,
                              child: CameraPreview(_cameraController!),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                  Positioned(
                    bottom: 16,
                    right: 16,
                    child: IconButton(
                      icon: Icon(Icons.switch_camera),
                      onPressed: _toggleCamera,
                      color: Colors.white,
                    ),
                  ),
                ],
              )
                  : Text('Initializing camera...'),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: getImageEmotion,
              child: Text('Select Image'),
            ),
            ElevatedButton(
              onPressed: predictLiveCamera,
              child: Text(isLiveCamera ? 'Stop Live Camera' : 'Start Live Camera'),
            ),
          ],
        ),
      ),
    );
  }
}
