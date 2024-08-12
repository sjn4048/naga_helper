keyboard_shortcut = '''
(function() {
    // 保存原始的 console.log 函数
    const originalConsoleLog = console.log;

    // 重写 console.log 函数
    console.log = function(message, ...optionalParams) {
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
                            selectElement.dispatchEvent(new Event('change', { bubbles: true }));
                            break;
                        }
                    }
                }
                }, 100);
        }
    };

    // 键盘事件监听器
    document.addEventListener('keydown', function(event) {
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
        switch(event.key) {
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
                selectElement.dispatchEvent(new Event('change', { bubbles: true }));
                break;
            case 's':
            case 'S':
            case 'ArrowDown':
                // 向下箭头，选择下一个选项
                selectElement.selectedIndex = (selectElement.selectedIndex + 1) % selectElement.length;
                // 手动触发 change 事件
                selectElement.dispatchEvent(new Event('change', { bubbles: true }));
                break;   
            default:
                break;
        }
    });
})();
'''