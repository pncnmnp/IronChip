# IronChip

An interpreter for CHIP-8 in Rust.

# Installation and Running

This interpreter runs directly in the terminal and uses certain [Kitty](https://sw.kovidgoyal.net/kitty/)-specific APIs for some of its functionalities. Therefore, it is necessary to install Kitty or a terminal emulator that is built on top of Kitty to run this interpreter.

You can find the executables to install Kitty in its [GitHub release page](https://github.com/kovidgoyal/kitty/releases).

To install the dependencies, clone this repository and run the following command in the root path:

```
cargo build
```

After that, any CHIP-8 ROM can be executed by running the following command along with its relative path:

```
cargo run <relative-path-to-rom>
```

# Screenshots

![image](https://github.com/pncnmnp/IronChip/assets/24948340/3d5277bd-912f-4559-887b-ba339165361a)

# Attribution

* The ROMs were taken from [David Matlack's Chip-8 emulator](https://github.com/dmatlack/chip8/tree/master/roms/games).
* Tian Yang's [Python-CHIP8-Emulator](https://github.com/AlpacaMax/Python-CHIP8-Emulator/tree/master) and Tobias Langhoff's [Guide to Making a CHIP-8 Emulator](https://tobiasvl.github.io/blog/write-a-chip-8-emulator/) were super helpful resources for this project.
* Tim Franssen's [chip8-test-suite](https://github.com/Timendus/chip8-test-suite) was a helpful resource for finding bugs. There are still a few more bugs left to squash :)

# License

The code is open-sourced under the MIT License.
