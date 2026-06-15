{
  description = "Python flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};

          python = pkgs.python314;
          pythonPackages = python.withPackages (
            ps: with ps; [
              pyside6
              matplotlib
              scipy
              numpy
              gmsh
            ]
          );
        in
        {
          default = pkgs.mkShell {
            buildInputs = [
              pythonPackages
            ];

            shellHook = ''
              python --version
            '';
          };
        }
      );
    };
}
