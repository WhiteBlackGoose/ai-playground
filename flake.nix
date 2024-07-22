{
  description = "Flake for this Playground";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  inputs.fenix.url = "github:nix-community/fenix";

  outputs = { nixpkgs, fenix, ... }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "aarch64-linux" "x86_64-linux" ];
    in
    {
      formatter.x86_64-linux = nixpkgs.legacyPackages.x86_64-linux.nixpkgs-fmt;
      devShells = nixpkgs.lib.genAttrs systems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          fx = fenix.packages.${system};
        in
        rec {
          default =
            pkgs.mkShell rec {
              nativeBuildInputs = with pkgs; [
                (fx.complete.withComponents [
                  "cargo"
                  "rustc"
                  "rust-src"
                  "rustfmt"
                  "clippy"
                  "rust-analyzer"
                  "miri"
                ])
                vscode-extensions.vadimcn.vscode-lldb
                cmake
                pkg-config
                fontconfig
                ninja
                gcc
                freetype
                graphene
                glib
                openssl
                bzip2

                cudaPackages.cudatoolkit
                cudaPackages.cudnn
                cudaPackages.cudnn.dev
                cudaPackages.libcublas
                cudaPackages.libcurand
                cudaPackages.libcufft
                clang
                libclang
                libGL
                xorg.libX11
                xorg.libXi
                xorg.libXcursor
                xorg.libXrandr
                libz libz.dev
                libxkbcommon
                wayland

                libv4l libv4l.dev
                opencv4
                linuxHeaders
              ] ++ (with pkgs.xorg; [
                libX11
                libX11.dev
                libXi
                libXcursor
                libXrandr
                libXft
                libXft.dev
                libXinerama
                stdenv.cc.cc.lib
              ]);
              VSCODE_CODELLDB = "${pkgs.vscode-extensions.vadimcn.vscode-lldb}";
              LD_LIBRARY_PATH = "/run/opengl-driver/lib:${ with pkgs; lib.makeLibraryPath
                nativeBuildInputs
              }";
              PKG_CONFIG_PATH = "${pkgs.gtk4.dev}/lib/pkgconfig";
              OPENSSL_LIB_DIR="${pkgs.openssl.out}/lib";
              OPENSSL_DIR="${pkgs.openssl.dev}";
            };
        });
    };
}
