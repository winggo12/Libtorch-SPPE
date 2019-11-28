// One-stop header.
#include <torch/script.h>

// headers for opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define kIMAGE_SIZE_H 224
#define kIMAGE_SIZE_W 224
#define kCHANNELS 3

bool LoadImage(std::string file_name, cv::Mat &image , cv::Mat &cloneimage) {
  image = cv::imread(file_name);  // CV_8UC3
  cloneimage = image.clone();
  if (image.empty() || !image.data) {
    return false;
  }
  cv::cvtColor(image, image, CV_BGR2RGB);
  std::cout << "== image size: " << image.size() << " ==" << std::endl;

  // scale image to fit
  cv::Size scale(kIMAGE_SIZE_W, kIMAGE_SIZE_H);
  cv::resize(image, image, scale);
  std::cout << "== simply resize: " << image.size() << " ==" << std::endl;

  // convert [unsigned int] to [float]
  image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

  return true;
}


int main(int argc, const char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: classifier <path-to-exported-script-module> "
              << std::endl;
    return -1;
  }

 torch::jit::script::Module module = torch::jit::load(argv[1]);
  std::cout << "== Switch to GPU mode" << std::endl;
  // to GPU
  module.to(at::kCUDA);

  //assert(module != nullptr);
  std::cout << "== PoseModel loaded!\n";

  std::string file_name = "";
  cv::Mat image;
  cv::Mat copied_image;
  while (true) {
    std::cout << "== Input image path: [enter Q to exit]" << std::endl;
    std::cin >> file_name;
    if (file_name == "Q") {
      break;
    }
    if (LoadImage(file_name, image ,copied_image)) {

      auto input_tensor = torch::from_blob(
          image.data, {1, kIMAGE_SIZE_W, kIMAGE_SIZE_H, kCHANNELS});
      input_tensor = input_tensor.permute({0, 3, 1, 2});
      input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
      input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
      input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);

      // to GPU
      input_tensor = input_tensor.to(at::kCUDA);

      auto output = module.forward({input_tensor}); //type : [ Variable[CUDAFloatType]{1,17,56,56} ]

      torch::Tensor out_tensor = output.toTensor();

      out_tensor = out_tensor.to(at::kCPU);


//-------------------Finding the keypoint with the highest probability -------------------------------------------//

      int coor[17][3];
      int max_x = 0;
      int max_y = 0;
      int prob = 0;

      float max = out_tensor[0][1][0][0].item().toFloat();

      for(int kpts=0;kpts<17;kpts++){
        max = out_tensor[0][kpts][0][0].item().toFloat();
       for(int i=0;i<56;i++){
          for(int j=0;j<56;j++){
            //std::cout << out_tensor[0][1][i][j].item().toFloat() << std::endl;
            if(out_tensor[0][kpts][i][j].item().toFloat() > max  ){
            //  std::cout << i << " " << j << " " ;
              max = out_tensor[0][kpts][i][j].item().toFloat();
              max_x = j ;
              max_y = i ;
              prob = (int)(max*100);
            }
          }
      }
              coor[kpts][0] = max_x ;
              coor[kpts][1] = max_y ;
              coor[kpts][2] = prob ;


      }

    

        //std::cout << out_tensor << std::endl; 

//-------------------Display Result -------------------------------------------// 

        cv::Point p(0,0);

        for(int kpts=0;kpts<17;kpts++){
        std::cout << "Keypoint : " << kpts << std::endl;
        p.x = (int)(((float)coor[kpts][0] * copied_image.size().width) / 56 ) ; 
        p.y = (int)(((float)coor[kpts][1] * copied_image.size().height) / 56 ) ;
        circle(copied_image, p, 10, cvScalar(0, 0, 255), - 1);

        std::cout << "x :" << coor[kpts][0] << " y : " << coor[kpts][1] << std::endl;
        std::cout << "Probability :" << coor[kpts][2] << std::endl;

      }

        std::cout << "Visualizing Result ..." << std::endl;

        imwrite( "../img/result.jpg", copied_image );

        /*imshow("Display", copied_image);
        
        if(cv::waitKey(0) == 'q'){

           cv::destroyWindow("Display");
        }      */


    } else {
      std::cout << "Can't load the image, please check your path." << std::endl;
    }
  }
}

