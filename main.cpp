#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <vulkan/vulkan.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <optional>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <set>
#include <map>
#include <unordered_map>

//#define NDEBUG

const int MAX_FRAMES_IN_FLIGHT = 2;

const uint32_t WIDTH  = 1440;
const uint32_t HEIGHT = 900;

const std::string MODEL_PATH = "../models/ivysaur.obj";
const std::string TEXTURE_PATH = "../textures/ivysaur_diffuse.jpg";

const std::vector<const char*> validationLayers = {
  "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// Create VkDebugUtilsMessengerEXT object
// by looking up the address with vkGetInstanceProcAddr
VkResult CreateDebugUtilsMessengerEXT(
  VkInstance instance,
  const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
  const VkAllocationCallbacks* pAllocator,
  VkDebugUtilsMessengerEXT* pDebugMessenger)
{
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
              vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(
  VkInstance instance,
  const VkDebugUtilsMessengerEXT debugMessenger,
  const VkAllocationCallbacks* pAllocator)
{
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
              vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;

  bool isComplete() {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};

struct SwapChainSupportDetails {
  vk::SurfaceCapabilitiesKHR        surfaceCapabilities;
  std::vector<vk::SurfaceFormatKHR> surfaceFormats;
  std::vector<vk::PresentModeKHR>   presentModes;
};

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  static vk::VertexInputBindingDescription getBindingDescription() {
    return vk::VertexInputBindingDescription(
      0, sizeof(Vertex), vk::VertexInputRate::eVertex
    );
  }

  static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
    std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions;

    // Position
    attributeDescriptions[0] = vk::VertexInputAttributeDescription(
      0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)
    );

    // Color
    attributeDescriptions[1] = vk::VertexInputAttributeDescription(
      1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)
    );

    // Texture coordinates
    attributeDescriptions[2] = vk::VertexInputAttributeDescription(
      2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord)
    );

    return attributeDescriptions;
  }

  // Operator to test for equality between vertces, used in hash function
  bool operator==(const Vertex& other) const {
    return pos == other.pos && color == other.color && texCoord == other.texCoord;
  }
};

// Hash function for vertices
namespace std {
  template<> struct hash<Vertex> {
    size_t operator()(Vertex const& vertex) const {
      return ((hash<glm::vec3>()(vertex.pos) ^
              (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
              (hash<glm::vec2>()(vertex.texCoord) << 1);
    }
  };
}

struct UniformBufferObject {
  // Account for Vulkan aligment requirements
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

class vkTriangleApplication {
public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  GLFWwindow*                    window;

  vk::Instance                   instance;
  vk::SurfaceKHR                 surface;
  VkDebugUtilsMessengerEXT       debugMessenger;

  vk::Device                     device;
  vk::PhysicalDevice             physicalDevice;

  vk::Queue                      graphicsQueue;
  vk::Queue                      presentQueue;

  vk::SwapchainKHR               swapChain;
  vk::Format                     swapChainImageFormat;
  vk::Extent2D                   swapChainExtent;
  std::vector<vk::Image>         swapChainImages;
  std::vector<vk::ImageView>     swapChainImageViews;
  std::vector<VkFramebuffer>     swapChainFramebuffers;

  vk::PipelineLayout             pipelineLayout;
  vk::PipelineCache              pipelineCache;
  vk::Pipeline                   graphicsPipeline;

  vk::DescriptorPool             descriptorPool;
  vk::DescriptorSetLayout        descriptorSetLayout;
  std::vector<vk::DescriptorSet> descriptorSets;

  vk::RenderPass                 renderPass;
  vk::CommandPool                commandPool;
  std::vector<vk::CommandBuffer> commandBuffers;

  std::vector<vk::Semaphore>     imageAvailableSemaphores;
  std::vector<vk::Semaphore>     renderFinishedSemaphores;
  std::vector<vk::Fence>         inFlightFences;
  std::vector<vk::Fence>         imagesInFlight;

  std::vector<uint32_t>          indices;
  std::vector<Vertex>            vertices;
  vk::Buffer                     indexBuffer;
  vk::Buffer                     vertexBuffer;
  vk::DeviceMemory               indexBufferMemory;
  vk::DeviceMemory               vertexBufferMemory;

  std::vector<vk::Buffer>        uniformBuffers;
  std::vector<vk::DeviceMemory>  uniformBuffersMemory;

  vk::Image                      textureImage;
  vk::DeviceMemory               textureImageMemory;
  vk::ImageView                  textureImageView;
  vk::Sampler                    textureSampler;

  vk::Image                      depthImage;
  vk::DeviceMemory               depthImageMemory;
  vk::ImageView                  depthImageView;

  // Render target
  vk::Image                      colorImage;
  vk::DeviceMemory               colorImageMemory;
  vk::ImageView                  colorImageView;

  vk::SampleCountFlagBits        msaaSamples = vk::SampleCountFlagBits::e1;

  size_t                         currentFrame = 0;
  bool                           framebufferResized = false;

  void initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Don't create OpenGL context
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(WIDTH, HEIGHT, "vkTriangle", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
  }

  static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = static_cast<vkTriangleApplication*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
  }

  void initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createCommandPool();
    createColorResources();
    createDepthResources();
    createFramebuffers();
    createTextureImage();
    createTextureImageView();
    createTextureImageSampler();
    loadModel();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      drawFrame();
    }

    device.waitIdle();
  }

  void cleanup() {
    cleanupSwapChain();

    for (size_t i=0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      device.destroySemaphore(imageAvailableSemaphores[i]);
      device.destroySemaphore(renderFinishedSemaphores[i]);
      device.destroyFence(inFlightFences[i]);
    }

    device.destroyDescriptorSetLayout(descriptorSetLayout);

    device.destroyBuffer(indexBuffer);
    device.freeMemory(indexBufferMemory);

    device.destroyBuffer(vertexBuffer);
    device.freeMemory(vertexBufferMemory);

    device.destroySampler(textureSampler);
    device.destroyImageView(textureImageView);
    device.destroyImage(textureImage);
    device.freeMemory(textureImageMemory);

    device.destroyCommandPool(commandPool);
    device.destroy();

    if (enableValidationLayers) {
      DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }

    instance.destroySurfaceKHR(surface);
    instance.destroy();

    glfwDestroyWindow(window);
    glfwTerminate();
  }

  void cleanupSwapChain() {
    device.destroyImageView(colorImageView);
    device.destroyImage(colorImage);
    device.freeMemory(colorImageMemory);

    device.destroyImageView(depthImageView);
    device.destroyImage(depthImage);
    device.freeMemory(depthImageMemory);

    for (auto framebuffer : swapChainFramebuffers) {
      device.destroyFramebuffer(framebuffer);
    }

    device.freeCommandBuffers(commandPool, commandBuffers);

    device.destroyDescriptorPool(descriptorPool);

    device.destroyPipeline(graphicsPipeline);
    device.destroyPipelineCache(pipelineCache);
    device.destroyPipelineLayout(pipelineLayout);
    device.destroyRenderPass(renderPass);

    for (auto imageView : swapChainImageViews) {
      device.destroyImageView(imageView);
    }

    for (size_t i=0; i < swapChainImages.size(); i++) {
      device.destroyBuffer(uniformBuffers[i]);
      device.freeMemory(uniformBuffersMemory[i]);
    }

    device.destroySwapchainKHR(swapChain);
  }


  void createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
      throw std::runtime_error("validation layers requested, but not available!");
    }

    vk::ApplicationInfo appInfo("vkTriangle", 1, "No Engine", 1, VK_API_VERSION_1_2);

    // Specify which global extensions and validation layers to use
    auto extensions = getRequiredExtensions();

    vk::InstanceCreateInfo createInfo(vk::InstanceCreateFlags(), &appInfo);
    createInfo.enabledExtensionCount   = extensions.size();
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (enableValidationLayers) {
      createInfo.enabledLayerCount   = validationLayers.size();
      createInfo.ppEnabledLayerNames = validationLayers.data();

      auto debugCreateInfo = getDebugMessengerCreateInfo();
      createInfo.pNext     = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
    } else {
      createInfo.enabledLayerCount = 0;

      createInfo.pNext = nullptr;
    }

    instance = vk::createInstance(createInfo);
  }

  VkDebugUtilsMessengerCreateInfoEXT getDebugMessengerCreateInfo() {
    // Specify types of severities for callback to be called for
    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
      | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
      | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
      | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo
    );

    // Specify message types that callback is notified for
    vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
      | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
      //| vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
    );

    return VkDebugUtilsMessengerCreateInfoEXT(
      vk::DebugUtilsMessengerCreateInfoEXT(
        {}, severityFlags, messageTypeFlags, &debugCallback
      )
    );
  }

  void setupDebugMessenger() {
    if (!enableValidationLayers) return;

    auto createInfo = getDebugMessengerCreateInfo();

    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger)
        != VK_SUCCESS)
    {
      throw std::runtime_error("failed to set up debug messenger!");
    }
  }

  void createSurface() {
    auto vSurface = VkSurfaceKHR(surface);

    if (glfwCreateWindowSurface(instance, window, nullptr, &vSurface) != VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }

    surface = vk::SurfaceKHR(vSurface);
  }

  void pickPhysicalDevice() {
    auto physicalDevices = instance.enumeratePhysicalDevices();

    if (physicalDevices.size() == 0) {
      throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    // Create map of physical devices sorted by rateDeviceSuitability()
    std::multimap<int, vk::PhysicalDevice> candidates;

    for (const auto& physicalDevice : physicalDevices) {
      int score = rateDeviceSuitability(physicalDevice);
      candidates.insert(std::make_pair(score, physicalDevice));
    }

    // Check if best candidate is suitable at all
    if (candidates.rbegin()->first > 0) {
      physicalDevice = candidates.rbegin()->second;
      msaaSamples = getMaxUsableSampleCount();
    } else {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
  }

  void createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::set<uint32_t> uniqueQueueFamilies = {
      indices.graphicsFamily.value(),
      indices.presentFamily.value()
    };

    // Select queue families to create
    float queuePriority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

    for (uint32_t queueFamily : uniqueQueueFamilies) {
      vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamily, 1, &queuePriority);
      queueCreateInfos.push_back(queueCreateInfo);
    }

    // Specify device features application will use
    vk::PhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    //deviceFeatures.sampleRateShading = VK_TRUE; // Enable sample shading (interior AA)

    device = physicalDevice.createDevice(
      vk::DeviceCreateInfo(
        {}, queueCreateInfos.size(), queueCreateInfos.data(),
        0, nullptr, // Enabled layers
        deviceExtensions.size(), deviceExtensions.data(),
        &deviceFeatures
      )
    );

    graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
    presentQueue  = device.getQueue(indices.presentFamily.value(), 0);
  }

  void createSwapChain() {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

    auto surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.surfaceFormats);
    auto presentMode   = chooseSwapPresentMode(swapChainSupport.presentModes);
    auto extent        = chooseSwapExtent(swapChainSupport.surfaceCapabilities);

    uint32_t imageCount = swapChainSupport.surfaceCapabilities.minImageCount + 1;
    if (  swapChainSupport.surfaceCapabilities.maxImageCount > 0
       && imageCount > swapChainSupport.surfaceCapabilities.maxImageCount
    ) {
      imageCount = swapChainSupport.surfaceCapabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo(
      {}, surface, imageCount,
      surfaceFormat.format, surfaceFormat.colorSpace, extent,
      1, vk::ImageUsageFlagBits::eColorAttachment
    );
    createInfo.preTransform = swapChainSupport.surfaceCapabilities.currentTransform;

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    uint32_t queueFamilyIndices[] = {
      indices.graphicsFamily.value(),
      indices.presentFamily.value()
    };

    if (indices.graphicsFamily != indices.presentFamily) {
      createInfo.imageSharingMode      = vk::SharingMode::eConcurrent;
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices   = queueFamilyIndices;
    } else {
      createInfo.imageSharingMode      = vk::SharingMode::eExclusive;
      createInfo.queueFamilyIndexCount = 0;
      createInfo.pQueueFamilyIndices   = nullptr;
    }

    swapChain            = device.createSwapchainKHR(createInfo);
    swapChainImages      = device.getSwapchainImagesKHR(swapChain);
    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent      = extent;
  }

  void recreateSwapChain() {
    // Handle widow minimization
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window, &width, &height);
      glfwWaitEvents();
    }

    device.waitIdle();

    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createColorResources();
    createDepthResources();
    createFramebuffers();
    createUniformBuffers();
    createDescriptorPool();
    createCommandBuffers();
  }

  vk::ImageView createImageView(
    vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags
  ) {
    auto imageView = device.createImageView(
      vk::ImageViewCreateInfo(
        {}, image, vk::ImageViewType::e2D, format,
        vk::ComponentMapping(vk::ComponentSwizzle::eIdentity),
        vk::ImageSubresourceRange(aspectFlags, 0, 1, 0, 1)
      )
    );

    return imageView;
  }

  void createImageViews() {
    swapChainImageViews.resize(swapChainImages.size());

    for (size_t i=0; i < swapChainImages.size(); i++) {
      swapChainImageViews[i] = createImageView(
        swapChainImages[i], swapChainImageFormat, vk::ImageAspectFlagBits::eColor
      );
    }
  }

  void createRenderPass() {
    vk::AttachmentDescription colorAttachment(
      {}, swapChainImageFormat,
      msaaSamples,
      vk::AttachmentLoadOp::eClear,
      vk::AttachmentStoreOp::eStore,
      vk::AttachmentLoadOp::eDontCare,
      vk::AttachmentStoreOp::eDontCare,
      vk::ImageLayout::eUndefined,
      vk::ImageLayout::eColorAttachmentOptimal
    );

    vk::AttachmentDescription depthAttachment(
      {}, findDepthFormat(),
      msaaSamples,
      vk::AttachmentLoadOp::eClear,
      vk::AttachmentStoreOp::eDontCare,
      vk::AttachmentLoadOp::eDontCare,
      vk::AttachmentStoreOp::eDontCare,
      vk::ImageLayout::eUndefined,
      vk::ImageLayout::eDepthStencilAttachmentOptimal
    );

    // Resolve the multiple fragments per pixel created via MSAA
    vk::AttachmentDescription colorAttachmentResolve(
      {}, swapChainImageFormat,
      vk::SampleCountFlagBits::e1,
      vk::AttachmentLoadOp::eDontCare,
      vk::AttachmentStoreOp::eStore,
      vk::AttachmentLoadOp::eDontCare,
      vk::AttachmentStoreOp::eDontCare,
      vk::ImageLayout::eUndefined,
      vk::ImageLayout::ePresentSrcKHR
    );

    vk::AttachmentReference colorAttachmentRef(
      0, vk::ImageLayout::eColorAttachmentOptimal
    );

    vk::AttachmentReference depthAttachmentRef(
      1, vk::ImageLayout::eDepthStencilAttachmentOptimal
    );

    vk::AttachmentReference colorAttachmentResolveRef(
      2, vk::ImageLayout::eColorAttachmentOptimal
    );

    vk::SubpassDescription subpass(
      {}, vk::PipelineBindPoint::eGraphics, 0,
      nullptr, 1,
      &colorAttachmentRef,
      &colorAttachmentResolveRef,
      &depthAttachmentRef
    );

    vk::SubpassDependency dependency(
      VK_SUBPASS_EXTERNAL, 0,
      vk::PipelineStageFlagBits::eColorAttachmentOutput,
      vk::PipelineStageFlagBits::eColorAttachmentOutput,
      vk::AccessFlagBits::eColorAttachmentWrite,
      vk::AccessFlagBits::eColorAttachmentWrite
    );

    std::array<vk::AttachmentDescription, 3> attachments = {
      colorAttachment,
      depthAttachment,
      colorAttachmentResolve
    };

    renderPass = device.createRenderPass(
      vk::RenderPassCreateInfo(
        {}, attachments.size(), attachments.data(),
        1, &subpass, 1, &dependency
      )
    );
  }

  void createDescriptorSetLayout() {
    vk::DescriptorSetLayoutBinding uboLayoutBinding(
      0, vk::DescriptorType::eUniformBuffer, 1,
      vk::ShaderStageFlagBits::eVertex, nullptr
    );

    vk::DescriptorSetLayoutBinding samplerLayoutBinding(
      1, vk::DescriptorType::eCombinedImageSampler, 1,
      vk::ShaderStageFlagBits::eFragment, nullptr
    );

    std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
      uboLayoutBinding,
      samplerLayoutBinding
    };

   descriptorSetLayout = device.createDescriptorSetLayout(
     vk::DescriptorSetLayoutCreateInfo({}, bindings.size(), bindings.data())
   );
  }

  void createGraphicsPipeline() {
    auto vertShaderCode = readFile("../shaders/shader.vert.spv");
    auto fragShaderCode = readFile("../shaders/shader.frag.spv");

    auto vertShaderModule = createShaderModule(vertShaderCode);
    auto fragShaderModule = createShaderModule(fragShaderCode);

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {
      vk::PipelineShaderStageCreateInfo(
        {}, vk::ShaderStageFlagBits::eVertex, vertShaderModule, "main"
      ),
      vk::PipelineShaderStageCreateInfo(
        {}, vk::ShaderStageFlagBits::eFragment, fragShaderModule, "main"
      )
    };

    // Fixed functions of graphics pipeline
    vk::Viewport viewport(
      0.0f, 0.0f,
      (float) swapChainExtent.width, (float) swapChainExtent.height,
      0.0f, 1.0f
    );
    vk::Rect2D scissor({0, 0}, swapChainExtent);

    vk::PipelineViewportStateCreateInfo viewportState({}, 1, &viewport, 1, &scissor);

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
      {}, vk::PrimitiveTopology::eTriangleList, VK_FALSE
    );

    auto bindingDescription    = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo(
      {}, 1, &bindingDescription,
      attributeDescriptions.size(),
      attributeDescriptions.data()
    );

    vk::PipelineRasterizationStateCreateInfo rasterizer(
      {}, VK_FALSE, VK_FALSE,
      vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack,
      vk::FrontFace::eCounterClockwise, VK_FALSE
    );
    rasterizer.lineWidth = 1.0f;

    vk::PipelineMultisampleStateCreateInfo multisampling(
      {}, msaaSamples,
      VK_FALSE, 1.0f, // Disable sample shading
      nullptr,
      VK_FALSE, VK_FALSE
    );

    vk::PipelineDepthStencilStateCreateInfo depthStencil(
      {}, VK_TRUE, VK_TRUE,
      vk::CompareOp::eLess,
      VK_FALSE, // depthBoundsTestEnable
      VK_FALSE  // stencilTestEnable
    );

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR
                                        | vk::ColorComponentFlagBits::eG
                                        | vk::ColorComponentFlagBits::eB
                                        | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = VK_FALSE;

    vk::PipelineColorBlendStateCreateInfo colorBlending(
      {}, VK_FALSE, vk::LogicOp::eClear, 1, &colorBlendAttachment
    );

    pipelineLayout = device.createPipelineLayout(
      vk::PipelineLayoutCreateInfo({}, 1, &descriptorSetLayout, 0, nullptr)
    );

    pipelineCache = device.createPipelineCache(vk::PipelineCacheCreateInfo());

    // Combine above structs to create graphics pipeline
    graphicsPipeline = device.createGraphicsPipeline(
      pipelineCache,
      vk::GraphicsPipelineCreateInfo(
        {},               shaderStages.size(), shaderStages.data(),
        &vertexInputInfo, &inputAssembly,      nullptr,
        &viewportState,   &rasterizer,         &multisampling,
        &depthStencil,    &colorBlending,      nullptr,
        pipelineLayout,   renderPass,          0
      )
    ).value;
    // Calling `.value` is a workaround for a known issue
    // Should be resolved after pull request #678 gets merged
    // https://github.com/KhronosGroup/Vulkan-Hpp/pull/678

    device.destroyShaderModule(vertShaderModule);
    device.destroyShaderModule(fragShaderModule);
  }

  void createFramebuffers() {
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (size_t i=0; i < swapChainImageViews.size(); i++) {
      std::array<vk::ImageView, 3> attachments = {
        colorImageView,
        depthImageView,
        swapChainImageViews[i],
      };

      swapChainFramebuffers[i] = device.createFramebuffer(
        vk::FramebufferCreateInfo(
          {}, renderPass,
          attachments.size(), attachments.data(),
          swapChainExtent.width, swapChainExtent.height, 1
        )
      );
    }
  }

  void createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    commandPool = device.createCommandPool(
      vk::CommandPoolCreateInfo({}, queueFamilyIndices.graphicsFamily.value())
    );
  }

  uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    auto memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i=0; i < memProperties.memoryTypeCount; i++) {
      if (   typeFilter & (1 << i)
          && (memProperties.memoryTypes[i].propertyFlags & properties) == properties
      ) {
        return i;
      }
    }

    throw std::runtime_error("failed to find suitable memory type!");
  }

  // Images can have different layouts that affect how the pixels are stored in memory
  // When performing operations on images, you want to transition the image
  // to a layout that is optimal for that operation's performance
  void transitionImageLayout(
    vk::Image image, vk::Format format,
    vk::ImageLayout oldLayout, vk::ImageLayout newLayout
  ) {
    auto cmd = beginSingleTimeCommands();

    vk::PipelineStageFlagBits srcPipelineStage;
    vk::PipelineStageFlagBits dstPipelineStage;

    vk::AccessFlagBits srcAccessMask;
    vk::AccessFlagBits dstAccessMask;

    // Specify source and destination pipeline barriers
    if (   oldLayout == vk::ImageLayout::eUndefined
        && newLayout == vk::ImageLayout::eTransferDstOptimal
    ) {
      srcAccessMask = static_cast<vk::AccessFlagBits>(0);
      dstAccessMask = vk::AccessFlagBits::eTransferWrite;

      srcPipelineStage = vk::PipelineStageFlagBits::eTopOfPipe;
      dstPipelineStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal
               && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal
    ) {
      srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      dstAccessMask = vk::AccessFlagBits::eShaderRead;

      srcPipelineStage = vk::PipelineStageFlagBits::eTransfer;
      dstPipelineStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
      throw std::invalid_argument("unsupported layout transition!");
    }

    vk::ImageMemoryBarrier barrier(
      srcAccessMask, dstAccessMask,
      oldLayout, newLayout,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
      image,
      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
    );

    cmd.pipelineBarrier(
      srcPipelineStage, dstPipelineStage,
      vk::DependencyFlagBits::eByRegion,
      0, nullptr,
      0, nullptr,
      1, &barrier
    );

    endSingleTimeCommands(cmd);
  }

  void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory) {
    buffer= device.createBuffer(
      vk::BufferCreateInfo({}, size, usage, vk::SharingMode::eExclusive)
    );

    auto memRequirements = device.getBufferMemoryRequirements(buffer);

    bufferMemory = device.allocateMemory(
      vk::MemoryAllocateInfo(
        memRequirements.size,
        findMemoryType(memRequirements.memoryTypeBits, properties)
      )
    );

    device.bindBufferMemory(buffer, bufferMemory, 0);
  }

  vk::CommandBuffer beginSingleTimeCommands() {
    auto commandBuffers = device.allocateCommandBuffers(
      vk::CommandBufferAllocateInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1)
    );

    vk::CommandBuffer& cmd = commandBuffers[0];

    cmd.begin(
      // eOneTimeSubmit specifies that each recording of the command buffer will
      // only be submitted once, and then reset and recorded again between submissions
      vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)
    );

    return cmd;
  }

  void endSingleTimeCommands(vk::CommandBuffer cmd) {
    cmd.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    graphicsQueue.submit(1, &submitInfo, nullptr);
    graphicsQueue.waitIdle();

    device.freeCommandBuffers(commandPool, 1, &cmd);
  }

  void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
    auto cmd = beginSingleTimeCommands();

    auto copyRegion = vk::BufferCopy(0, 0, size);
    cmd.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(cmd);
  }

  void copyBufferToImage(
    vk::Buffer buffer,
    vk::Image image,
    uint32_t width, uint32_t height
  ) {
    auto cmd = beginSingleTimeCommands();

    // Specify which part of the buffer is to be copied to which part of the image
    vk::BufferImageCopy copyRegion(
      0, 0, 0,
      vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
      vk::Offset3D(0, 0, 0),
      vk::Extent3D(width, height, 1)
    );

    cmd.copyBufferToImage(
      buffer, image,
      vk::ImageLayout::eTransferDstOptimal,
      1, &copyRegion
    );

    endSingleTimeCommands(cmd);
  }

  template <typename T>
  void createVkBuffer(const std::vector<T>& data, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory, vk::BufferUsageFlagBits usage) {
    vk::DeviceSize bufferSize = sizeof(data[0]) * data.size();

    // Staging buffer is on the CPU
    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;

    createBuffer(
      bufferSize,
      vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible
      | vk::MemoryPropertyFlagBits::eHostCoherent,
      stagingBuffer, stagingBufferMemory
    );

    void* memData = device.mapMemory(stagingBufferMemory, 0, bufferSize);
    memcpy(memData, data.data(), (size_t) bufferSize);
    device.unmapMemory(stagingBufferMemory);

    // Create buffer on the GPU
    createBuffer(
      bufferSize,
      vk::BufferUsageFlagBits::eTransferDst | usage,
      vk::MemoryPropertyFlagBits::eDeviceLocal,
      buffer, bufferMemory
    );

    // Copy staging buffer to GPU buffer before creation is complete
    copyBuffer(stagingBuffer, buffer, bufferSize);

    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingBufferMemory);
  }

  void createImage(
    uint32_t width, uint32_t height,
    vk::SampleCountFlagBits numSamples,
    vk::Format format,
    vk::ImageTiling tiling,
    vk::ImageUsageFlags usage,
    vk::MemoryPropertyFlagBits properties,
    vk::Image& image, vk::DeviceMemory& imageMemory
  ) {
    image = device.createImage(
      vk::ImageCreateInfo(
        {}, vk::ImageType::e2D, format,
        vk::Extent3D(width, height, 1), 1, 1,
        numSamples,
        tiling, usage,
        vk::SharingMode::eExclusive
      )
    );

    auto memRequirements = device.getImageMemoryRequirements(image);

    imageMemory = device.allocateMemory(
      vk::MemoryAllocateInfo(
        memRequirements.size,
        findMemoryType(memRequirements.memoryTypeBits, properties)
      )
    );

    device.bindImageMemory(image, imageMemory, 0);
  }

  void createTextureImage() {
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(
      TEXTURE_PATH.c_str(),
      &texWidth, &texHeight, &texChannels,
      STBI_rgb_alpha
    );

    vk::DeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
      throw std::runtime_error("failed to load texture image!");
    }

    // Staging buffer is on the CPU
    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;

    createBuffer(
      imageSize,
      vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible
      | vk::MemoryPropertyFlagBits::eHostCoherent,
      stagingBuffer, stagingBufferMemory
    );

    void* data = device.mapMemory(stagingBufferMemory, 0, imageSize);
    memcpy(data, pixels, (size_t) imageSize);
    device.unmapMemory(stagingBufferMemory);

    stbi_image_free(pixels);

    createImage(
      texWidth, texHeight,
      vk::SampleCountFlagBits::e1,  // TODO
      vk::Format::eR8G8B8A8Srgb,
      vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst
      | vk::ImageUsageFlagBits::eSampled,
      vk::MemoryPropertyFlagBits::eDeviceLocal,
      textureImage, textureImageMemory
    );

    transitionImageLayout(
      textureImage, vk::Format::eR8G8B8A8Srgb,
      vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal
    );

    copyBufferToImage(stagingBuffer, textureImage, texWidth, texHeight);

    transitionImageLayout(
      textureImage, vk::Format::eR8G8B8A8Srgb,
      vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal
    );

    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingBufferMemory);
  }

  void createTextureImageView() {
    textureImageView = createImageView(
      textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor
    );
  }

  void createTextureImageSampler() {
    textureSampler = device.createSampler(
      vk::SamplerCreateInfo(
        // Specify how to interpolate texels that are
        // magnified (oversampling) or minified (undersampling)
        {}, vk::Filter::eLinear, vk::Filter::eLinear,
        vk::SamplerMipmapMode::eLinear,
        // Sampler address mode specifies how to
        // read texels that are outside of the image
        vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode::eRepeat,
        vk::SamplerAddressMode::eRepeat,
        0.0f,
        VK_TRUE, 16.0f, // Anisotropic filtering
        VK_FALSE, vk::CompareOp::eAlways,
        0.0f, 0.0f
      )
    );
  }

  void loadModel() {
    tinyobj::attrib_t attrib;  // Holds positions, normals, and texture coordinates
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(
          &attrib, &shapes, &materials,
          &warn, &err,
          MODEL_PATH.c_str()
        )
    ) {
      throw std::runtime_error(warn + err);
    }

    std::unordered_map<Vertex, uint32_t> uniqueVertices{};

    // Iterate over the shapes vertices and add them to the
    // class memember `vertices` vector
    for (const auto& shape : shapes) {
      for (const auto& index : shape.mesh.indices) {
        Vertex vertex{};

        // Load arrays of floats into vectors
        vertex.pos = {
          attrib.vertices[3 * index.vertex_index + 0],
          attrib.vertices[3 * index.vertex_index + 1],
          attrib.vertices[3 * index.vertex_index + 2]
        };

        vertex.texCoord = {
              attrib.texcoords[2 * index.texcoord_index + 0],
          1.0-attrib.texcoords[2 * index.texcoord_index + 1] // Flip vertical coordinate
        };

        vertex.color = {1.0f, 1.0f, 1.0f};

        if (uniqueVertices.count(vertex) == 0) {
          uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
          vertices.push_back(vertex);
        }

        indices.push_back(uniqueVertices[vertex]);
      }
    }
  }

  void createVertexBuffer() {
    createVkBuffer<Vertex>(
      vertices, vertexBuffer, vertexBufferMemory,
      vk::BufferUsageFlagBits::eVertexBuffer
    );
  }

  void createIndexBuffer() {
    createVkBuffer<uint32_t>(
      indices, indexBuffer, indexBufferMemory,
      vk::BufferUsageFlagBits::eIndexBuffer
    );
  }

  void createDescriptorPool() {
    vk::DescriptorPoolSize uniformBufferPoolSize(
      vk::DescriptorType::eUniformBuffer, swapChainImages.size()
    );

    vk::DescriptorPoolSize samplerPoolSize(
      vk::DescriptorType::eCombinedImageSampler, swapChainImages.size()
    );

    std::array<vk::DescriptorPoolSize, 2> poolSizes = {
      uniformBufferPoolSize,
      samplerPoolSize
    };

    descriptorPool = device.createDescriptorPool(
      vk::DescriptorPoolCreateInfo(
        {}, swapChainImages.size(),
        poolSizes.size(), poolSizes.data()
      )
    );
  }

  void createDescriptorSets() {
    descriptorSets.resize(swapChainImages.size());
    std::vector<vk::DescriptorSetLayout> layouts(
      swapChainImages.size(), descriptorSetLayout
    );

    descriptorSets = device.allocateDescriptorSets(
      vk::DescriptorSetAllocateInfo(
        descriptorPool, swapChainImages.size(), layouts.data()
      )
    );

    for (size_t i=0; i < swapChainImages.size(); i++) {
      vk::DescriptorBufferInfo bufferInfo(
        uniformBuffers[i], 0, sizeof(UniformBufferObject)
      );

      vk::DescriptorImageInfo imageInfo(
        textureSampler, textureImageView,
        vk::ImageLayout::eShaderReadOnlyOptimal
      );

      vk::WriteDescriptorSet uniformBufferDescriptorWrite(
        descriptorSets[i], 0, 0, 1,
        vk::DescriptorType::eUniformBuffer,
        nullptr, &bufferInfo, nullptr
      );

      vk::WriteDescriptorSet samplerDescriptorWrite(
        descriptorSets[i], 1, 0, 1,
        vk::DescriptorType::eCombinedImageSampler,
        &imageInfo, nullptr, nullptr
      );

      std::array<vk::WriteDescriptorSet, 2> descriptorWrites = {
        uniformBufferDescriptorWrite,
        samplerDescriptorWrite
      };

      device.updateDescriptorSets(
        descriptorWrites.size(),
        descriptorWrites.data(),
        0, nullptr
      );
    }
  }

  void createUniformBuffers() {
    vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(swapChainImages.size());
    uniformBuffersMemory.resize(swapChainImages.size());

    for (size_t i=0; i < swapChainImages.size(); i++) {
      createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eUniformBuffer,
          vk::MemoryPropertyFlagBits::eHostVisible
        | vk::MemoryPropertyFlagBits::eHostCoherent,
        uniformBuffers[i], uniformBuffersMemory[i]
      );
    }
  }

  void updateUniformBuffer(uint32_t currentImage) {
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
      currentTime - startTime
    ).count();

    UniformBufferObject ubo{};

    glm::mat4 uboRotation = glm::rotate(
      glm::mat4(1.0f),
      (1.0f + time / 2.0f) * glm::radians(90.0f),
      glm::vec3(0.0f, 1.0f, 0.0f) // Rotation axis
    );

    glm::mat4 uboTranslation = glm::translate(
        glm::mat4(1.0f),
        glm::vec3(0.0f, -0.67f, 0.0f)
    );

    ubo.model = uboRotation * uboTranslation;

    ubo.view = glm::lookAt(
      glm::vec3(0.0f, 1.0f, 3.0f), // Eye position
      glm::vec3(0.0f),            // Center position
      glm::vec3(0.0f, 1.0f, 0.0f) // Up Axis
    );

    ubo.proj = glm::perspective(
      glm::radians(45.0f),
      swapChainExtent.width / (float) swapChainExtent.height,
      0.1f, 10.0f
    );

    // Invert Y axis to acccount for difference between OpenGL and Vulkan
    ubo.proj[1][1] *= -1;

    void* data = device.mapMemory(uniformBuffersMemory[currentImage], 0, sizeof(ubo));
    memcpy(data, &ubo, sizeof(ubo));
    device.unmapMemory(uniformBuffersMemory[currentImage]);
  }

  void createCommandBuffers() {
    commandBuffers.resize(swapChainFramebuffers.size());

    commandBuffers = device.allocateCommandBuffers(
      vk::CommandBufferAllocateInfo(
        commandPool, vk::CommandBufferLevel::ePrimary, commandBuffers.size()
      )
    );

    for (size_t i=0; i < commandBuffers.size(); i++) {
      vk::CommandBuffer& cmd = commandBuffers[i];

      std::array<vk::ClearValue, 2> clearValues{
        vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}),
        vk::ClearDepthStencilValue(1.0f, 0)
      };

      cmd.begin(vk::CommandBufferBeginInfo());

      cmd.beginRenderPass(
        vk::RenderPassBeginInfo(
          renderPass,
          swapChainFramebuffers[i],
          vk::Rect2D({0, 0}, swapChainExtent),
          clearValues.size(), clearValues.data()
        ),
        vk::SubpassContents::eInline
      );

      cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

      vk::Buffer vertexBuffers[] = {vertexBuffer};
      vk::DeviceSize offsets[] = {0};

      cmd.bindVertexBuffers(0, 1, vertexBuffers, offsets);
      cmd.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);

      cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        pipelineLayout, 0, 1,
        &descriptorSets[i], 0, nullptr
      );

      cmd.drawIndexed(indices.size(), 1, 0, 0, 0);

      cmd.endRenderPass();
      cmd.end();
    }
  }

  void createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight.resize(swapChainImages.size());

    for (size_t i=0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      imageAvailableSemaphores[i]
        = device.createSemaphore(vk::SemaphoreCreateInfo(), nullptr);

      renderFinishedSemaphores[i]
        = device.createSemaphore(vk::SemaphoreCreateInfo(), nullptr);

      inFlightFences[i]
        = device.createFence(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled), nullptr);
    }
  }

  void drawFrame() {
    device.waitForFences(1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    vk::Result result = device.acquireNextImageKHR(
      swapChain, UINT64_MAX,
      imageAvailableSemaphores[currentFrame],
      nullptr, &imageIndex
    );

    // Usually happens after a window resize
    if (   result == vk::Result::eErrorOutOfDateKHR
        || result == vk::Result::eSuboptimalKHR
    ) {
      recreateSwapChain();
      return;
    }

    updateUniformBuffer(imageIndex);

    // Check if a previous frame is using this image
    // (i.e. there is its fence to wait on)
    if (imagesInFlight[imageIndex]) {
      device.waitForFences(1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
    }
    // Mark the image as now being in use by this frame
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];

    vk::Semaphore signalSemaphores[]    = {renderFinishedSemaphores[currentFrame]};
    vk::Semaphore waitSemaphores[]      = {imageAvailableSemaphores[currentFrame]};
    vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};

    vk::SubmitInfo submitInfo(
      1, waitSemaphores, waitStages,
      1, &commandBuffers[imageIndex],
      1, signalSemaphores
    );

    device.resetFences(1, &inFlightFences[currentFrame]);

    graphicsQueue.submit(1, &submitInfo, inFlightFences[currentFrame]);

    vk::SwapchainKHR swapChains[] = {swapChain};

    result = presentQueue.presentKHR(
      vk::PresentInfoKHR(1, signalSemaphores, 1, swapChains, &imageIndex)
    );

    if (   result == vk::Result::eErrorOutOfDateKHR
        || result == vk::Result::eSuboptimalKHR
        || framebufferResized
    ) {
      recreateSwapChain();
      return;
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  int rateDeviceSuitability(vk::PhysicalDevice physicalDevice) {
    vk::PhysicalDeviceFeatures deviceFeatures = physicalDevice.getFeatures();
    vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();

    int score = 0;

    // Specify deviceProperties that are preferred
    score += deviceProperties.limits.maxImageDimension2D;
    if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
      score += 1000;
    }

    // Specify deviceFeatures and extension support that application requires
    if (   !deviceFeatures.shaderInt16
        || !deviceFeatures.samplerAnisotropy
        || !checkDeviceExtensionSupport(physicalDevice)
    ) {
      return 0;
    }

    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
    if (   swapChainSupport.surfaceFormats.empty()
        || swapChainSupport.presentModes.empty()
    ) {
      return 0;
    }

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    if (!indices.isComplete()) {
      return 0;
    }

    return score;
  }

  void createColorResources() {
    vk::Format colorFormat = swapChainImageFormat;

    createImage(
      swapChainExtent.width, swapChainExtent.height,
      msaaSamples,
      colorFormat,
      vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransientAttachment
      | vk::ImageUsageFlagBits::eColorAttachment,
      vk::MemoryPropertyFlagBits::eDeviceLocal,
      colorImage, colorImageMemory
    );

    colorImageView = createImageView(
      colorImage, colorFormat, vk::ImageAspectFlagBits::eColor
    );
  }

  void createDepthResources() {
    auto depthFormat = findDepthFormat();

    createImage(
      swapChainExtent.width, swapChainExtent.height,
      msaaSamples,
      depthFormat,
      vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eDepthStencilAttachment,
      vk::MemoryPropertyFlagBits::eDeviceLocal,
      depthImage, depthImageMemory
    );

    depthImageView = createImageView(
      depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth
    );
  }

  bool hasStencilComponent(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint
        || format == vk::Format::eD24UnormS8Uint;
  }

  vk::Format findSupportedFormat(
    const std::vector<vk::Format>& candidates,
    vk::ImageTiling tiling, vk::FormatFeatureFlags features
  ) {
    for (auto format : candidates) {
      auto props = physicalDevice.getFormatProperties(format);

      if (   tiling == vk::ImageTiling::eLinear
          && (props.linearTilingFeatures & features) == features
      ) {
        return format;
      } else if (   tiling == vk::ImageTiling::eOptimal
                 && (props.optimalTilingFeatures & features) == features
      ) {
        return format;
      }
    }

    throw std::runtime_error("failed to find supported format!");
  }

  vk::Format findDepthFormat() {
    return findSupportedFormat(
      {
        vk::Format::eD32Sfloat,
        vk::Format::eD32SfloatS8Uint,
        vk::Format::eD24UnormS8Uint
      }, vk::ImageTiling::eOptimal,
      vk::FormatFeatureFlagBits::eDepthStencilAttachment
    );
  }

  vk::SampleCountFlagBits getMaxUsableSampleCount() {
    auto deviceProperties = physicalDevice.getProperties();

    vk::SampleCountFlags counts = deviceProperties.limits.framebufferColorSampleCounts
                                 & deviceProperties.limits.framebufferDepthSampleCounts;
    if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
    if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
    if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
    if (counts & vk::SampleCountFlagBits::e8)  { return vk::SampleCountFlagBits::e8;  }
    if (counts & vk::SampleCountFlagBits::e4)  { return vk::SampleCountFlagBits::e4;  }
    if (counts & vk::SampleCountFlagBits::e2)  { return vk::SampleCountFlagBits::e2;  }

    return vk::SampleCountFlagBits::e1;
  }

  SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice physicalDevice) {
    SwapChainSupportDetails details;
    details.surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
    details.surfaceFormats      = physicalDevice.getSurfaceFormatsKHR(surface);
    details.presentModes        = physicalDevice.getSurfacePresentModesKHR(surface);

    return details;
  }

  vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> availableFormats) {
    // Prefer SRGB color format if available
    for (const auto& availableFormat : availableFormats) {
      if (  availableFormat.format     == vk::Format::eB8G8R8A8Srgb
         && availableFormat.colorSpace == vk::ColorSpaceKHR::eVkColorspaceSrgbNonlinear
      ) {
        return availableFormat;
      }
    }

    return availableFormats[0];
  }

  vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
    // Prefer triple buffering if available
    for (const auto& availablePresentMode : availablePresentModes) {
      if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
        return availablePresentMode;
      }
    }

    return vk::PresentModeKHR::eFifo;
  }

  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
    // Note: Swap extent is the resolution of the swap chain images.
    //       Almost always equal to the resolution of the window being drawn to.

    // Window managers that allow extent to differ from resolution
    // of the window being drawn to set currentExtent to UINT32_MAX
    if (capabilities.currentExtent.width != UINT32_MAX) {
      return capabilities.currentExtent;
    } else {
      int width, height;
      glfwGetFramebufferSize(window, &width, &height);

      vk::Extent2D actualExtent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height)
      };

      // Clamp value of WIDTH and HEIGHT between the min and max
      // extents supported by the implementation
      actualExtent.width = std::max(capabilities.minImageExtent.width,
                                    std::min(capabilities.maxImageExtent.width,
                                             actualExtent.width));

      actualExtent.height = std::max(capabilities.minImageExtent.height,
                                     std::min(capabilities.maxImageExtent.height,
                                              actualExtent.height));

      return actualExtent;
    }
  }

  bool checkDeviceExtensionSupport(vk::PhysicalDevice physicalDevice) {
    auto availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();

    std::set<std::string> requiredExtensions(
      deviceExtensions.begin(), deviceExtensions.end()
    );

    for (const auto& extension : availableExtensions) {
      requiredExtensions.erase(static_cast<std::string>(extension.extensionName));
    }

    return requiredExtensions.empty();
  }

  QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice physicalDevice) {
    QueueFamilyIndices indices;
    auto queueFamilies = physicalDevice.getQueueFamilyProperties();

    // Prefer a queue family with both graphics capabilities and surface support
    // Otherwise track the indices of the two different queue families that, together,
    // support both of these things
    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
      if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
        indices.graphicsFamily = i;
      }

      vk::Bool32 presentSupport = false;
      presentSupport = physicalDevice.getSurfaceSupportKHR(i, surface);

      if (presentSupport) {
        indices.presentFamily = i;
      }

      if (indices.isComplete()) {
        break;
      }

      i++;
    }

    return indices;
  }

  std::vector<const char*> getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(
      glfwExtensions, glfwExtensions + glfwExtensionCount
    );

    if (enableValidationLayers) {
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
  }

  bool checkValidationLayerSupport() {
    auto availableLayers = vk::enumerateInstanceLayerProperties();

    // Check if all of the layers in validationLayers exist in availableLayers
    for (const char* layerName : validationLayers) {
      bool layerFound = false;

      for (const auto& layerProperties : availableLayers) {
        if (strcmp(layerName, layerProperties.layerName) == 0) {
          layerFound = true;
          break;
        }
      }

      if (!layerFound) {
        return false;
      }
    }

    return true;
  }

  static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData
  ) {
    std::cerr << std::endl << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
  }

  static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
  }

  vk::ShaderModule createShaderModule(const std::vector<char>& code) {
    return device.createShaderModule(
      vk::ShaderModuleCreateInfo(
        {}, code.size(), reinterpret_cast<const uint32_t*>(code.data())
      )
    );
  }
};

int main() {
  vkTriangleApplication app;

  try {
    app.run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
