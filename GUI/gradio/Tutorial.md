# gradio教程

内容来源于官网中的快速教程：https://gradio.app/quickstart/

## [快速开始 - Quickstart](https://gradio.app/quickstart/)

先决条件：Gradio需要Python 3.7或更高版本，仅此而已！

与他人分享你的机器学习模型、API或者数据科学工作流程的最佳方式之一就是建立一个交互式的应用程序，gradio就是一个这样的工具，全部使用python，只需要几行代码，就可以快速的构建一个演示应用并分享它们。

### 安装方式

```bash
pip install gradio
```

基础示例：hello world，这是示例演示了gradio的基础操作，通常gradio会缩写为gr以便于使用，将主界面赋值给demo，通过launch开始webui。Interface会定义一个最初级的界面框架，包含处理函数、输入、输出三个部分。

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

demo.launch()   
```

![hello_world](images/1.1.hello_world.png)

> 在使用脚本模式的时候，界面需要通过在浏览器中访问本地端口http://localhost:7860来查看，如pycharm。在Jupyter Notebook中，这个界面会自动的出现。

在本地开发时，如果您想将代码作为Python脚本运行，可以使用Gradio CLI以重载模式启动应用程序，这将提供无缝快速的开发。在《自动重新加载指南》中了解有关重新加载的详细信息。

```bash
# 重载模式 - 快速开发
gradio app.py
# 注意 python app.py 的形式并不提供自动重载的机制。
```

### [`Interface`](https://gradio.app/quickstart/#the-interface-class) 类

Interface类可以用用户接口包装任何Python函数。在上面的例子中，我们看到了一个简单的基于文本的函数，但该函数可以是任何东西，从音乐生成器到税收计算器，再到预训练的机器学习模型的预测函数。Interface类使用三个参数进行初始化：

- `fn`: 包装UI的函数
- `inputs`: 用于输入的组件（例如“文本”、“图像”或“音频”）
- `outputs`: 用于输出的组件（例如“文本”、“图像”或“标签”）

### 组件属性 - Components Attributes

如果需要更加定制化的内容，比如在Textbox中加入输入的提示，可以自定义input。

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(lines=2, placeholder="Name Here..."),
    outputs="text",
)
demo.launch()
```

![组件属性](images/1.2.components_attributes)

### 多个输入和输出的组件 - Multiple Input and Output Components

假如你有一个更复杂的函数，有多个输出和输入。可以通过传递输入和输出的列表，并定义一个能够处理多输入和多输出的函数来实现。

```python
import gradio as gr

def greet(name, is_morning, temperature):
    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation} {name}. It is {temperature} degrees today"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "checkbox", gr.Slider(0, 100)],
    outputs=["text", "number"],
)
demo.launch()
```

### 一个图片的例子 - An Image Example

输入数据为图片时的一个示例，当使用图片组件作为输入的时候，函数会获得一个numpy array形式的矩阵，维度为(高, 宽, 3) ，最后一个维度是RGB。

```python
import numpy as np
import gradio as gr

def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189], 
        [0.349, 0.686, 0.168], 
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

demo = gr.Interface(sepia, gr.Image(shape=(200, 200)), "image")
demo.launch()
```

### Blocks：块，更加灵活和可控 - More Flexibility and Control

Gradio提供了两个类来构建应用程序：

1. **Interface**：一种高级的、抽象的结构，之前的示例均在使用。
2. **Blocks**：一种低级API，用于设计web app，更灵活的布局和数据流程。Blocks允许你做一些像是多个数据流和功能的演示、控制组件在页面上的显示位置、处理复杂的数据流程（例如，输出可以作为其他功能的输入），以及基于用户交互更新组件的属性/可见性。如果您需要这种可定制性，请尝试Blocks！

下面是一个简单的示例：

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Greet")
    greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet")


demo.launch()
```

![hello_block](images/1.3.hello_block)

> 需要注意：
>
> - Block是用with子句生成的，在该子句中创建的任何组件都会自动添加到应用程序中。
> - 组件按创建顺序垂直显示在应用程序中。（也可以自定义布局！）
> - 创建一个button按钮，然后向按钮添加了click点击事件侦听器。与Interface类似，click方法的输入需要Python函数、输入组件和输出组件。

下面来展示一个Blocks的无线可能。这里将翻转数字和翻转图片的功能同时进行了实现，并且相互之间不会冲突，按需使用。

```python
import numpy as np
import gradio as gr


def flip_text(x):
    return x[::-1]


def flip_image(x):
    return np.fliplr(x)


with gr.Blocks() as demo:
    gr.Markdown("Flip text or image files using this demo.")
    with gr.Tab("Flip Text"):
        text_input = gr.Textbox()
        text_output = gr.Textbox()
        text_button = gr.Button("Flip")
    with gr.Tab("Flip Image"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("Flip")

    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at me...")

    text_button.click(flip_text, inputs=text_input, outputs=text_output)
    image_button.click(flip_image, inputs=image_input, outputs=image_output)

demo.launch()
```

恭喜你，你现在已经熟悉了Gradio的基本知识！如果感兴趣的话，可以继续阅读并了解更多的内容，这个教程中的所有内容都来自于官方文档，这里只是一个中文化的罗列、叙述和小结，最新的内容[访问这里](https://www.gradio.app/)来查看。

## 关键特征 - Key Features

在对主体进行总结之前，这个部分总结了一些gradio一些特性

1. [Adding example inputs](### example-inputs)
2. [Passing custom error messages](https://gradio.app/key-features/#errors)
3. [Adding descriptive content](https://gradio.app/key-features/#descriptive-content)
4. [Setting up flagging](https://gradio.app/key-features/#flagging)
5. [Preprocessing and postprocessing](https://gradio.app/key-features/#preprocessing-and-postprocessing)
6. [Styling demos](https://gradio.app/key-features/#styling)
7. [Queuing users](https://gradio.app/key-features/#queuing)
8. [Iterative outputs](https://gradio.app/key-features/#iterative-outputs)
9. [Progress bars](https://gradio.app/key-features/#progress-bars)
10. [Batch functions](https://gradio.app/key-features/#batch-functions)
11. [Running on collaborative notebooks](https://gradio.app/key-features/#colab-notebooks)

### 示例输入 - example-inputs

在**Interface**组件中，可以提供一些供用户进行选择的一些示例，当使用`examples=`关键字时，可以提供一个嵌套的列表，外层列表代表了一个数据样本，内层代表每个组件的一个输入。可以通过该文档[Docs](https://gradio.app/docs#components)来查看不同组件的输入格式要求。

```python
import gradio as gr

def calculator(num1, operation, num2):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        if num2 == 0:
            raise gr.Error("Cannot divide by zero!")
        return num1 / num2

demo = gr.Interface(
    calculator,
    [
        "number", 
        gr.Radio(["add", "subtract", "multiply", "divide"]),
        "number"
    ],
    "number",
    examples=[
        [5, "add", 3],
        [4, "divide", 2],
        [-4, "multiply", 2.5],
        [0, "subtract", 1.2],
    ],
    title="Toy Calculator",
    description="Here's a sample toy calculator. Allows you to calculate things like $2+2=4$",
)
demo.launch()
```

![示例输入](images/2.1.example_inputs)

### 错误提示 - Errors

您希望将自定义错误消息传递给用户。要执行此操作，请raise gr.Error("自定义消息")以显示错误消息。如果你试图在上面的计算器演示中除以零，弹出模式将显示自定义错误消息。在文档中了解有关错误的详细信息。

### 描述性的内容 - Descriptive Content

在**Interface**中，使用`title=`和`description=`等关键词可以生成描述性的内容，来帮助用户理解你的app中想要表达的意思。有以下三个参数：

- `title`: 它接受**文本**并可以在界面的最顶部显示，还可以成为页面标题。
- `description`: 它接受text、markdown或HTML并将其放在标题下。
- `article`: 它接受text、markdown或HTML 并将其放置在interface窗口的最下面。

![interface中的描述性组件](images/2.2.Descriptive_content_in_interface)

在**Blocks**中可以使用`gr.Markdown(...)`或`gr.HTML(...)`组件来插入text、markdown或HTML。

另一个有用的关键字参数是`label=`，它存在于每个组件中。这将修改每个零部件顶部的标签文本。您还可以将`info=`关键字参数添加到**Textbox**或**Radio**等表单元素中，以提供有关其用法的更多信息。

```python
gr.Number(label='Age', info='In years, must be greater than 0')
```

### 标记 - Flagging

**Interface**中默认会存在"Flag"按钮，当用户测试app时，出现导致错误或者意外的输入时，可以将这些输入标记并保存以便查看。保存目录由`flagging_dir=`提供，CSV文件将记录标记的输入。如果接口涉及文件数据，例如图像和音频组件，则会创建文件夹来存储这些标记的数据。下面是一个示例：

```directory
+-- flagged/
|   +-- logs.csv
|   +-- im/
|   |   +-- 0.png
|   |   +-- 1.png
|   +-- Output/
|   |   +-- 0.png
|   |   +-- 1.png
```

*flagged/logs.csv*

```csv
im,Output
im/0.png,Output/0.png
im/1.png,Output/1.png
```

如果你想让用户提供一个标记的理由，你可以向**Interface**的**flagging_options**参数传递一个字符串列表。用户在标记时必须选择其中一个字符串，这将被保存为CSV的一个附加列。

### 预处理和后处理 - preprocessing and postprocessing

![annotated](images/2.3.dataflow.svg)

正如您所看到的，Gradio包括了很多可以处理各种不同数据类型的组件，如图像、音频和视频。大多数组件既可以用作输入，也可以用作输出。
当一个组件被用作输入时，Gradio会自动预处理，将数据从用户浏览器发送的类型（如网络摄像头快照的base64表示）转换为函数可以接受的形式（如numpy数组）。
类似地，当组件用作输出时，Gradio会自动处理将函数返回的数据（如图像路径列表）转换为可以在用户浏览器中显示的表单（如base64格式的图像库）所需的后处理。
在构造图像组件

```

```

时，可以使用参数控制预处理。例如，在这里，如果您使用以下参数实例化Image组件，它将把图像转换为PIL类型，并将其重塑为（100，100），而不管它提交的原始大小是什么：

```
img = gr.Image(shape=(100, 100), type="pil")
```

相反，在这里，我们保持图像的原始大小，但在将其转换为numpy数组之前反转颜色：

```
img = gr.Image(invert_colors=True, type="numpy")
```

后期处理要容易得多！Gradio自动识别返回数据的格式（例如，图像是numpy数组还是str文件路径？），并将其后处理为浏览器可以显示的格式。
更多关于每个组件的所有预处理相关参数，可查看官方文档[Docs](https://gradio.app/docs) 。

### 自定义风格 - styling

使用Gradio的主题是自定义应用程序外观的最简单方法。您可以从各种主题中进行选择，也可以创建自己的主题。为此，请将`theme=`传递给**Interface**函数。例如：

```python
demo = gr.Interface(..., theme=gr.themes.Monochrome())
```

Gradio提供了一组预构建的主题，您可以从`gr.themes.*`加载这些主题。您可以扩展这些主题或从头开始创建自己的主题。更多详细信息，请参阅主题指南[Theming guide](https://gradio.app/theming-guide)。
为了获得额外的样式功能，您可以使用`CSS＝`将任何CSS传递给您的应用程序。Gradio应用程序的基类是Gradio容器，因此下面是一个更改Gradio应用的背景颜色的示例：

```
with gr.Interface(css=".gradio-container {background-color: red}") as demo:
    ...
```

一些组件可以通过style（）方法进行额外的样式设置。例如：

```
img = gr.Image("lion.jpg").style(height='24', rounded=False)
```

更多相关的信息，可查看官方文档[Docs](https://gradio.app/docs) 。

### 队列 - Queuing

> 这个技术是针对于需要在线访问的情况而定的，相当于有多个人在访问服务器。

如果您的应用程序预计会有大量流量，请使用queue()方法来控制处理速率。这将使调用排队，因此一次只处理一定数量的请求。排队使用websocket，这也可以防止网络超时，因此如果函数的推理时间很长（>1min），则应该使用排队。

`Interface`:

```
demo = gr.Interface(...).queue()
demo.launch()
```

`Blocks`:

```
with gr.Blocks() as demo:
    #...
demo.queue()
demo.launch()
```

您可以控制一次处理的请求数，如下所示：

```
demo.queue(concurrency_count=3)
```

关于配置其他排队参数，请参阅排队文档[Docs on queueing](https://gradio.app/docs/#queue)。

要仅指定某些函数用于在块中排队，请执行以下操作：

```
with gr.Blocks() as demo2:
    num1 = gr.Number()
    num2 = gr.Number()
    output = gr.Number()
    gr.Button("Add").click(
        lambda a, b: a + b, [num1, num2], output)
    gr.Button("Multiply").click(
        lambda a, b: a * b, [num1, num2], output, queue=True)
demo2.launch()
```

### 迭代输出 - Iterative Output

在某些情况下，您可能希望持续输出一系列数据，而不是一次显示单个输出。例如，您可能有一个图像生成模型，并且希望显示在每个步骤生成的图像，直到生成最终图像。或者你可能有一个聊天机器人，它一次只流式传输一个单词的响应，而不是一次全部返回。

这种情况下，你可以使用生成器函数来替代常规的函数。通常yield被放置在某种循环中，下面是一个生成器的例子。

```python
def my_generator(x):
    for i in range(x):
        yield i
```

接下来，这里有一个（假）图像生成模型，它在输出图像之前生成几个步骤的噪声：

```python
import gradio as gr
import numpy as np
import time

# define core fn, which returns a generator {steps} times before returning the image
def fake_diffusion(steps):
    for _ in range(steps):
        time.sleep(1)
        image = np.random.random((600, 600, 3))
        yield image
    image = "https://gradio-builds.s3.amazonaws.com/diffusion_image/cute_dog.jpg"
    yield image


demo = gr.Interface(fake_diffusion, inputs=gr.Slider(1, 10, 3), outputs="image")

# define queue - required for generators
demo.queue()

demo.launch()
```

> 我们添加了在迭代器中加入了**time.sleep(1)**来创建了一个人工的暂停，这样我们就能观察到不同步之下，迭代器的变化了。
>
> 此外，向gradio提供生成器需要在底层接口或块中**启用queuing**

### 进度条 - Progress Bars

Gradio支持字创建自定义进度条的功能，这样就可以向用户显示进度更新。启用此功能时，需要在方法中添加一个**gr.Progress**的实例，之后你可以直接使用0-1之间的浮点数调用这个实例，或者使用Process中的**tqdm()**来跟踪一个迭代进度。如下图琐事，queue功能也必须启用。

```python
import gradio as gr
import time

def slowly_reverse(word, progress=gr.Progress()):
    progress(0, desc="Starting")
    time.sleep(1)
    progress(0.05)
    new_string = ""
    for letter in progress.tqdm(word, desc="Reversing"):
        time.sleep(0.25)
        new_string = letter + new_string
    return new_string

demo = gr.Interface(slowly_reverse, gr.Text(), gr.Text())

if __name__ == "__main__":
    demo.queue(concurrency_count=10).launch()
```

> 如果使用**tqdm**库，您甚至可以通过将默认参数设置为**gr.progress（track_tqdm=True）**，从函数中已经存在的任何tqdm.tqdm自动报告进度更新！

### 批处理函数 - Batch Functions

Gradio支持传递批处理函数的功能。批处理函数只是接收输入列表并返回预测列表的函数。

例如，这里有一个批处理函数，它接受两个输入列表（单词列表和int列表），并返回一个修剪后的单词列表作为输出：

```python
import time

def trim_words(words, lens):
    trimmed_words = []
    time.sleep(5)
    for w, l in zip(words, lens):
        trimmed_words.append(w[:int(l)])        
    return [trimmed_words]
```

使用批处理函数的优点是，如果您启用**queuing**，Gradio服务器可以自动批处理传入请求并并行处理它们，这可能会加快您的演示速度。下面是Gradio代码的示例（注意**batch=True和max_batch_size=16**——这两个参数都可以传递到事件触发器或Interface类中）

```python
demo = gr.Interface(trim_words, ["textbox", "number"], ["output"], 
                    batch=True, max_batch_size=16)
demo.queue()
demo.launch()
```

```python
import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        word = gr.Textbox(label="word")
        leng = gr.Number(label="leng")
        output = gr.Textbox(label="Output")
    with gr.Row():
        run = gr.Button()

    event = run.click(trim_words, [word, leng], output, batch=True, max_batch_size=16)

demo.queue()
demo.launch()
```

在上面的例子中，可以并行处理16个请求。许多Hugging Face中的transformer和diffusers模型都可以和Gradio中的batch模型自然的工作。

> 注意：在Gradio中使用批处理函数需要在底层接口或块中启用排队（请参阅上面的排队部分）。

### Colab笔记本 - Colab Notebook
Gradio可以在任何运行Python的地方运行，包括本地jupyter笔记本电脑以及协作笔记本电脑，如[Google Colab](https://colab.research.google.com/)。在本地jupyter笔记本电脑和Google Colab笔记本电脑的情况下，Gradio运行在本地服务器上，您可以在浏览器中与之交互。（注意：对于Google Colab，这是通过服务人员隧道 [service worker tunneling](https://github.com/tensorflow/tensorboard/blob/master/docs/design/colab_integration.md)实现，这需要在浏览器中启用cookie。）对于其他远程笔记本电脑，Gradio也将在服务器上运行，但您需要使用SSH隧道在本地浏览器中查看应用程序。通常，一个更简单的选择是使用Gradio的内置公共链接，这将在[下一个指南中讨论](https://www.gradio.app/sharing-your-app/#sharing-demos). 。

## 分享你的APP -  Sharing Your App

# 构建接口 - Building Interfaces

# 构建块 - Building with Blocks

# 构建其他框架 - Integrating Other Frameworks

# 表格数据科学与绘图 - Tabular Data Science And Plots

## 

# 客户端库 - Client Libraries

# 其他教程 - Other Tutorials

