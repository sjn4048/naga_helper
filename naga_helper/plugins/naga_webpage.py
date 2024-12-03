keyboard_shortcut = '''
(function () {
    // 保存原始的 console.log 函数
    const originalConsoleLog = console.log;

    // 重写 console.log 函数
    console.log = function (message, ...optionalParams) {
        // 调用原始的 console.log 函数
        originalConsoleLog.apply(console, [message, ...optionalParams]);

        // 检查是否输出了 'finished'
        if (typeof message === 'string' && message.includes('総局数')) {
            setTimeout(() => {
                // 获取 URL 中的参数
                const urlParams = new URLSearchParams(window.location.search);
                const mainParam = urlParams.get('main');
                // 获取 <select> 元素
                const selectElement = document.querySelector('#app > div > div > div:nth-child(1) > div.column.is-one-quarter > div.columns > div > div:nth-child(2) > select');
                if (mainParam && selectElement) {
                    // 查找 <option> 元素中 text 等于 mainParam 的选项
                    for (let i = 0; i < selectElement.options.length; i++) {
                        if (selectElement.options[i].text === mainParam) {
                            // 设置新的选中项
                            selectElement.selectedIndex = i;
                            // 手动触发 change 事件
                            selectElement.dispatchEvent(new Event('change', {bubbles: true}));
                            break;
                        }
                    }
                }
                // 创建Note div
                if (!document.getElementById('HelperNote')) {
                    // 获取目标元素
                    const targetElement = document.querySelector('#app > div > div > div:nth-child(1) > div.column.is-one-quarter > div.columns > div > div:nth-child(5)');

                    const newParagraph = document.createElement('p');
                    newParagraph.id = 'HelperNote'
                    newParagraph.textContent = ''; // 设置要展示的文字内容
                    newParagraph.style.textAlign = 'left'; // 设置左端对齐
                    newParagraph.style.display = 'block'; // 设置左端对齐
                    newParagraph.style.fontSize = '24px'; // 设置左端对齐
                    newParagraph.style.fontFamily = 'sans-serif';
                    // 将新元素插入到目标元素之后
                    targetElement.parentNode.insertBefore(newParagraph, targetElement.nextSibling);
                }
            }, 100);
        }
    };

    // 回调函数的占位符，需要用户自己实现
    async function submitNote(raw_name, user_id, junmu, text) {
        console.log('Note submitted:', raw_name, user_id, junmu, text);
        // 构造请求数据
        const data = {
            raw_name: raw_name,
            junmu: junmu.join(),
            text: text,
            user_id: user_id // 假设前端有用户ID，如果没有则需要从后端获取
        };

        try {
            // 发送POST请求到后端
            const response = await fetch('/api/naga/note_report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                body: new URLSearchParams(data).toString()
            });

            // 检查响应状态
            if (!response.ok) {
                alert('提交失败');
            }

            // 解析响应数据
            const result = await response.json();
            if (result.status === 1) {
                HelperNotes.push({
                    junmu: junmu.join(),
                    note: text,
                    raw_name: raw_name,
                    timestamp: 0,
                    user_id: user_id,
                });
                document.getElementById("HelperNote").textContent = text
                alert('提交成功');
            } else {
                alert(result.msg);
            }
        } catch (error) {
            console.error('提交失败:', error);
            alert('提交失败: ' + error.message);
        }
    }

    function removeElements(...elements) {
        elements.forEach(element => {
            if (element && element.remove) {
                element.remove();
            }
        });
    }

    let inNote = false;

    // 键盘事件监听器
    document.addEventListener('keydown', function (event) {
        if (inNote) {
            return
        }
        // 获取前、后、last error、next error按钮的元素
        const prevButton = document.querySelector('#app > div > div > div:nth-child(1) > div.column.is-one-quarter > div.columns > div > div.columns.is-multiline.is-mobile > button:nth-child(3)');
        const nextButton = document.querySelector('#app > div > div > div:nth-child(1) > div.column.is-one-quarter > div.columns > div > div.columns.is-multiline.is-mobile > button:nth-child(4)');
        const lastErrorButton = document.querySelector('#app > div > div > div:nth-child(1) > div.column.is-one-quarter > div.columns > div > div.columns.is-multiline.is-mobile > button:nth-child(5)');
        const nextErrorButton = document.querySelector('#app > div > div > div:nth-child(1) > div.column.is-one-quarter > div.columns > div > div.columns.is-multiline.is-mobile > button:nth-child(6)');
        const lastGameButton = document.querySelector('#app > div > div > div:nth-child(1) > div.column.is-one-quarter > div.columns > div > div.columns.is-multiline.is-mobile > button:nth-child(1)');
        const nextGameButton = document.querySelector('#app > div > div > div:nth-child(1) > div.column.is-one-quarter > div.columns > div > div.columns.is-multiline.is-mobile > button:nth-child(2)');
        const selectElement = document.querySelector('#app > div > div > div:nth-child(1) > div.column.is-one-quarter > div.columns > div > div:nth-child(5) > select');
        if (!selectElement) {
            console.error('Select element not found');
            return;
        }

        switch (event.key) {
            case 'a':
            case 'A':
            case 'ArrowLeft':
                if (prevButton) {
                    prevButton.click();
                }
                break;
            case 'd':
            case 'D':
            case 'ArrowRight':
                if (nextButton) {
                    nextButton.click();
                }
                break;
            case 'q':
            case 'Q':
                if (lastErrorButton) {
                    lastErrorButton.click();
                }
                break;
            case 'e':
            case 'E':
                if (nextErrorButton) {
                    nextErrorButton.click();
                }
                break;
            case 'z':
            case 'Z':
                if (lastGameButton) {
                    lastGameButton.click();
                }
                break;
            case 'c':
            case 'C':
                if (nextGameButton) {
                    nextGameButton.click();
                }
                break;
            case 'w':
            case 'W':
            case 'ArrowUp':
                // 向上箭头，选择上一个选项
                selectElement.selectedIndex = (selectElement.selectedIndex - 1 + selectElement.length) % selectElement.length;
                // 手动触发 change 事件
                selectElement.dispatchEvent(new Event('change', {bubbles: true}));
                break;
            case 's':
            case 'S':
            case 'ArrowDown':
                // 向下箭头，选择下一个选项
                selectElement.selectedIndex = (selectElement.selectedIndex + 1) % selectElement.length;
                // 手动触发 change 事件
                selectElement.dispatchEvent(new Event('change', {bubbles: true}));
                break;
            case 'n':
            case 'N':
                // 从URL中获取raw_name和user_id参数
                inNote = true
                const urlParams = new URLSearchParams(window.location.search);
                const raw_name = urlParams.get('raw');
                const user_id = urlParams.get('user_id');
                const junmu = window.junmu || 'UNKNOWN';

                // 添加输入框和按钮到页面并居中
                const container = document.createElement('div');
                container.style.position = 'fixed';
                container.style.top = '50%';
                container.style.left = '50%';
                container.style.transform = 'translate(-50%, -50%)';
                container.style.display = 'flex';
                container.style.justifyContent = 'center';
                container.style.alignItems = 'center';
                container.style.backgroundColor = 'white';
                container.style.padding = '10px';
                container.style.borderRadius = '8px';
                container.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';

                // 创建输入框
                const input = document.createElement('input');
                input.type = 'text';
                input.placeholder = '请输入批注';
                input.style.width = '300px';
                input.style.padding = '5px';
                input.style.border = '1px solid #ccc';
                input.style.borderRadius = '4px';
                input.style.fontFamily = 'sans-serif';

                // 自动补全
                const datalist = document.createElement('datalist');
                datalist.id = 'noteSuggestions';
                input.setAttribute('list', 'noteSuggestions');

                NoteStrings.forEach(function(suggestion) {
                  const option = document.createElement('option');
                  option.value = suggestion;
                  datalist.appendChild(option);
                });

                // 创建提交按钮
                const submitButton = document.createElement('button');
                submitButton.textContent = '提交';
                submitButton.style.marginLeft = '10px';
                submitButton.style.padding = '5px 10px';
                submitButton.style.border = 'none';
                submitButton.style.backgroundColor = '#2891ff';
                submitButton.style.color = 'white';
                submitButton.style.borderRadius = '4px';
                submitButton.style.cursor = 'pointer';
                submitButton.style.fontFamily = 'sans-serif';

                // 创建关闭按钮
                const closeButton = document.createElement('button');
                closeButton.textContent = '关闭';
                closeButton.style.marginLeft = '10px';
                closeButton.style.padding = '5px 10px';
                closeButton.style.border = 'none';
                closeButton.style.backgroundColor = '#ff2135';
                closeButton.style.color = 'white';
                closeButton.style.borderRadius = '4px';
                closeButton.style.cursor = 'pointer';
                closeButton.style.fontFamily = 'sans-serif';

                // 点击提交按钮时触发回调函数
                submitButton.addEventListener('click', function () {
                    submitNote(raw_name, user_id, junmu, input.value).then(() => {
                        removeElements(input, submitButton, closeButton, container);
                        inNote = false;
                    })
                });

                input.addEventListener('keypress', function (e) {
                    if (e.key === 'Enter') {
                        submitNote(raw_name, user_id, junmu, input.value).then(() => {
                            removeElements(input, submitButton, closeButton, container);
                            inNote = false;
                        })
                    }
                });

                // 点击关闭按钮时移除元素
                closeButton.addEventListener('click', function () {
                    removeElements(input, submitButton, closeButton, container);
                    inNote = false;
                });

                // 按Esc键关闭输入框
                document.addEventListener('keydown', function closeOnEsc(e) {
                    if (e.key === 'Escape') {
                        removeElements(input, submitButton, closeButton, container);
                        document.removeEventListener('keydown', closeOnEsc);
                        inNote = false;
                    }
                });
                container.appendChild(datalist);
                container.appendChild(input);
                container.appendChild(submitButton);
                container.appendChild(closeButton);

                document.body.append(container);
                input.focus(); // 自动聚焦到输入框
                event.preventDefault(); // 阻止默认行为，防止输入框中出现N
                break;
            default:
                break;
        }
    });
})();
'''
