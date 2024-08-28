#!/usr/share/miniconda2/envs/py39/bin/python3

import os
import curses
import re
import pdb

def debug_print(*args, sep=' ', end='\n'):
    """将内容写入到指定的文件中"""
    with open("hardy.log", 'a') as file:
        file.write(sep.join(map(str, args)) + end)

def proc_file_read_token(proc_file, token_regex):
    try:
        with open(proc_file, 'r') as f:
            lines = f.readlines()
            pattern = re.compile(token_regex)
            total_value = 0

            for line in lines:
                key, value = line.split(':', 1)
                key = key.strip()
                if pattern.match(key):
                    total_value += int(value.split()[0]);

            return total_value

    except FileNotFoundError:
        print(f"Error: {proc_file} not found.")

    return 0

def proc_file_read(proc_file):
    value = ""
    try:
        with open(proc_file, 'r') as f:
            value = f.read().replace('\0', ' ').strip()
    except FileNotFoundError:
        print(f"Error: /proc/{pid}/cmdline not found.")
    except PermissionError:
        print(f"Error: Permission denied to access /proc/{pid}/cmdline.")

    return value

def get_meminfo_item(x):
    value = proc_file_read_token("/proc/meminfo", x)
    if x == "MemTotal":
        return {"item": x, "value": value, "next": None}
    elif x == "MemFree":
        return {"item": x, "value": value, "next": None}
    elif x == "Buffers":
        return {"item": x, "value": value, "next": None}
    elif x == "Cached":
        return {"item": x, "value": value, "next": None}
    elif x == "AnonPages":
        return {"item": x, "value": value, "next": menu_process_rss_anon}
    elif x == "Slab":
        return {"item": x, "value": value, "next": None}
    elif x == "KernelStack":
        return {"item": x, "value": value, "next": None}
    elif x == "PageTables":
        return {"item": x, "value": value, "next": None}
    elif x == "Percpu":
        return {"item": x, "value": value, "next": None}
    elif x == "Hugetlb":
        return {"item": x, "value": value, "next": None}
    else:
        raise Exception(f"Unknown token {x} in file /proc/meminfo")

top_menu_meminfo = {
        "menu_name": "meminfo",
        "get_menu_func": lambda : [ "MemTotal", "MemFree", "Buffers", "Cached", "AnonPages", "Slab", "KernelStack", "PageTables", "Percpu", "Hugetlb"],
        "get_item_func": get_meminfo_item,
        "sort_item_func": None,
        "text_header_func": lambda :   f"Type"[:16].ljust(16) + "    " + f"Size"[:32].ljust(32),
        "text_item_func": lambda item: f"{item['item'][:16].ljust(16)}    {str(item['value'])[:32].ljust(32)}",
        "sub_item_func":  lambda item: menu_process_rss_anon if item['item'] == "AnonPages" else None
}

menu_process_rss_anon = {
        "menu_name": "proc_anon",
        "get_menu_func": lambda : [entry for entry in os.listdir('/proc') if entry.isdigit()],
        "get_item_func": lambda pid: {
                            "pid": pid,
                            "cmd": proc_file_read(f"/proc/{pid}/cmdline"),
                            "memory": proc_file_read_token(f"/proc/{pid}/status", "RssAnon")
                        },
        "text_item_func": lambda item: f"{item['pid'][:16].ljust(16)} {item['cmd'][:64].ljust(64)}    {str(item['memory'])[:32].ljust(32)}",
        "text_header_func": lambda :   f"pid"[:16].ljust(16) + " " + f"cmd"[:64].ljust(64) + "    " + f"Size"[:32].ljust(32),
        "sort_item_func": lambda _list : _list.sort(key= lambda x: x['memory'], reverse=True),
        "sub_item_func":  lambda item: None,
}

def generate_menu(menu_define):
    menu_data = []
    get_menu_func = menu_define["get_menu_func"]
    text_item_func = menu_define["text_item_func"]
    sub_item_func = menu_define["sub_item_func"]
    text_header_func = menu_define["text_header_func"]

    if get_menu_func is not None:
        menu_list = get_menu_func()
        get_item_func = menu_define["get_item_func"]
        sort_item_func = menu_define["sort_item_func"]
        for menu in menu_list:
            menu_data.append(get_item_func(menu))

        if sort_item_func is not None:
            sort_item_func(menu_data)

    else:
       pass

    header_line = "  " + text_header_func()
    lines = [f"+ {text_item_func(item)}" if sub_item_func(item) is not None else f"  {text_item_func(item)}"
             for item in menu_data]
    sub_menu_define = [sub_item_func(item) for item in menu_data]
    return (lines, sub_menu_define, header_line)

def main_menu(stdscr):
    """Main menu for the memory analyzer tool"""
    curses.curs_set(0)  # 隐藏光标
    curses.start_color()  # 启用颜色支持
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_YELLOW)
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_BLUE)

    menu_stack = []
    menu_stack.append(generate_menu(top_menu_meminfo))
    current_row = 0
    pad_start_line = 0
    (win_height, win_width) = stdscr.getmaxyx()
    debug_print("win_height", win_height, "win_width", win_width)

    while True:
        stdscr.clear()  # 清除 stdscr 内容
        stdscr.refresh()  # 刷新 stdscr 来显示新的内容

        # fetch the top menu
        (menu_lines, sub_menu, header_line) = menu_stack[-1]


        pad_height = len(menu_lines) + 1    # plus a header line
        pad_width = max(len(row) for row in menu_lines) + 2
        pad = curses.newpad(pad_height, pad_width)

        pad.clear()
        pad.attron(curses.color_pair(2))
        pad.addstr(0, 0, header_line)
        pad.attroff(curses.color_pair(2))

        for idx, row in enumerate(menu_lines):
            x = 0
            y = idx + 1

            if idx < pad_start_line:
                continue

            if idx > pad_start_line + win_height:
                break

            debug_print("print", idx, "start", pad_start_line, "cusor", current_row)
            if idx == current_row:
                pad.attron(curses.color_pair(1))
                pad.addstr(y, x, row)
                pad.attroff(curses.color_pair(1))
            else:
                pad.addstr(y, x, row)

        pad.refresh(0, 0, 0, 0, win_height - 1, win_width - 1)
        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
            if current_row == pad_start_line:
                pad_start_line -= 1

        elif key == curses.KEY_DOWN:
            if current_row - pad_start_line < win_width:
                current_row += 1
            else:
                pad_start_line += 1
                current_row += 1

        elif key == curses.KEY_ENTER or key in [10, 13]:
            if sub_menu[current_row] is not None:
                menu_stack.append(generate_menu(sub_menu[current_row]))
                current_row = 0
                pad_start_line = 0
def main():
    curses.wrapper(main_menu)

if __name__ == "__main__":
    main()
